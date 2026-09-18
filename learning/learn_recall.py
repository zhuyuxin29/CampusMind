# BM25 和向量两路都使用 LangChain 检索器
import os
import hashlib
import logging
from pathlib import Path

import jieba
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 隐藏 jieba 初始化时的调试输出
jieba.setLogLevel(logging.WARNING)

# 1. 配置路径
project_dir = Path(r"D:\VsCodeProjects\CampusMind")
file_path = project_dir / "data" / "campus_text.pdf"
db_path = project_dir / "data" / "chroma_db"

# 2. 读取硅基流动密钥
load_dotenv(project_dir / ".env")
api_key = os.getenv("SILICONFLOW_API_KEY")

if not api_key:
    raise ValueError("请检查 .env 中的 SILICONFLOW_API_KEY")

if not file_path.is_file():
    raise FileNotFoundError(f"找不到 PDF：{file_path}")

# 3. 加载 PDF
loader = PyPDFLoader(str(file_path))
pages = loader.load()

# 4. 切分文本
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)

chunks = text_splitter.split_documents(pages)
chunks = [chunk for chunk in chunks if chunk.page_content.strip()]

if not chunks:
    raise ValueError("没有提取到文字，请检查 PDF 是否为扫描件")

# 5. 创建嵌入模型客户端
embeddings_model = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
    check_embedding_ctx_length=False,
    chunk_size=16,
)

# 6. 创建或打开本地 Chroma 数据库
vector_store = Chroma(
    collection_name="campus_bge_m3_v1",
    embedding_function=embeddings_model,
    persist_directory=str(db_path),
)

# 7. 为片段生成固定 ID
chunk_ids = []

for i, chunk in enumerate(chunks):
    chunk.metadata = {
        key: value
        for key, value in chunk.metadata.items()
        if isinstance(value, (str, int, float, bool))
    }
    chunk.metadata["chunk_index"] = i

    identity = (
        f"{file_path.resolve()}\n"
        f"{chunk.metadata.get('page', 0)}\n"
        f"{i}\n"
        f"{chunk.page_content}"
    )

    chunk_id = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()

    chunk_ids.append(chunk_id)

# 8. 分批入库，跳过已经存在的片段
added_count = 0
skipped_count = 0
batch_size = 16

for start in range(0, len(chunks), batch_size):
    batch_chunks = chunks[start:start + batch_size]
    batch_ids = chunk_ids[start:start + batch_size]

    existing = vector_store.get(ids=batch_ids, include=[])
    existing_ids = set(existing["ids"])

    new_chunks = []
    new_ids = []

    for chunk, chunk_id in zip(batch_chunks, batch_ids):
        if chunk_id in existing_ids:
            skipped_count += 1
        else:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if new_chunks:
        # 自动生成向量，同时保存正文和元数据
        vector_store.add_documents(
            documents=new_chunks,
            ids=new_ids,
        )
        added_count += len(new_chunks)

print(
    f"入库完成：新增 {added_count} 个片段，"
    f"跳过 {skipped_count} 个已有片段。"
)

# 9. 从数据库读取正文，为 BM25 准备资料
stored = vector_store.get(
    include=["documents", "metadatas"],
)

documents = [
    Document(
        page_content=text,
        metadata=metadata or {},
    )
    for text, metadata in zip(
        stored["documents"],
        stored["metadatas"],
    )
    if text and text.strip()
]

if not documents:
    raise ValueError("数据库中没有可检索的文本")

# 10. 创建 BM25 检索器
candidate_k = min(3, len(documents))  # 每路最多召回 3 条
final_k = 2                          # 融合后最多返回 2 条

bm25_retriever = BM25Retriever.from_documents(
    documents,
    # 中文分词；lambda 是这里使用的简短匿名函数
    preprocess_func=lambda text: [
        word.lower()
        for word in jieba.lcut(text)
        if word.strip() and any(char.isalnum() for char in word)
    ],
    k=candidate_k,
)

# 11. 将 Chroma 包装成向量检索器
vector_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": candidate_k},
)

# 12. 输入问题，两路分别召回
query = input("\n请输入问题：").strip()

if not query:
    raise ValueError("问题不能为空")

bm25_results = bm25_retriever.invoke(query)
vector_results = vector_retriever.invoke(query)

# 13. RRF 排名融合：去重，并累加两路排名贡献
scores = {}
doc_map = {}

for result_list in [bm25_results, vector_results]:
    for rank, doc in enumerate(result_list, start=1):
        # 用来源、位置和正文识别同一个片段
        key = (
            doc.metadata.get("source", ""),
            doc.metadata.get("page", -1),
            doc.metadata.get("chunk_index", -1),
            doc.page_content,
        )

        doc_map[key] = doc
        scores[key] = scores.get(key, 0) + 0.5 / (60 + rank)

ranked_keys = sorted(
    scores,
    key=scores.get,
    reverse=True,
)

hybrid_results = [
    doc_map[key]
    for key in ranked_keys[:final_k]
]

# 14. 只打印混合召回的最终结果
print("\n===== 混合召回结果 =====")

for i, doc in enumerate(hybrid_results, start=1):
    print(f"\n--- 第 {i} 条 ---")
    print(doc.page_content)

    source = doc.metadata.get("source", "未知来源")
    page = doc.metadata.get("page")
    page_number = page + 1 if isinstance(page, int) else "未知"

    print(f"来源：{source}")
    print(f"PDF 页码：{page_number}")