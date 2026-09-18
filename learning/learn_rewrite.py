# 查询重写
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

# 12. 接收原始问题
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

query = input("\n请输入问题：").strip()

if not query:
    raise ValueError("问题不能为空")

# 13. 创建用于查询改写的 DeepSeek 客户端
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

if not deepseek_api_key:
    raise ValueError("请检查 .env 中的 DEEPSEEK_API_KEY")

rewrite_llm = ChatOpenAI(
    model="deepseek-flash",
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com",
    temperature=0,
    max_tokens=500,
    timeout=60,
    max_retries=1,
    extra_body={"thinking": {"type": "disabled"}},
)

# 要求输出 JSON，便于程序读取
rewrite_client = rewrite_llm.bind(
    response_format={"type": "json_object"}
)

# 14. 根据原始问题，生成两条不同措辞的查询
rewrite_response = rewrite_client.invoke([
    SystemMessage(
        content=(
            "你负责为校园知识库生成检索查询。\n"
            "根据用户原始问题，生成两条有助于检索的中文改写。\n"
            "一条使用自然语言表达，另一条使用简洁的检索关键词。\n"
            "保留原始问题中的身份、名称、编号、时间、否定和限制条件。\n"
            "不要猜测答案，不要增加原问题没有要求的主题。\n"
            "不要回答用户问题，只返回 JSON。\n"
            '格式：{"queries": ["改写问题1", "改写问题2"]}'
        )
    ),
    HumanMessage(content=query),
])

# 15. 解析改写结果，始终保留原始问题
search_queries = [query]

try:
    rewrite_data = json.loads(rewrite_response.content)
    rewritten_queries = rewrite_data["queries"]

    if not isinstance(rewritten_queries, list):
        raise ValueError("queries 应当是列表")

    for rewritten in rewritten_queries:
        if not isinstance(rewritten, str):
            continue

        rewritten = rewritten.strip()

        # 去掉空查询、重复查询，只保留最多两条改写
        if rewritten and rewritten not in search_queries:
            search_queries.append(rewritten)

        if len(search_queries) >= 3:
            break

except (ValueError, KeyError, TypeError):
    # 格式异常时，仍可以用原始问题继续召回
    print("改写结果格式异常，本次仅使用原始问题。")

# 练习阶段保留这个打印，检查问题是否被改偏
print("\n===== 实际使用的查询 =====")

for i, search_query in enumerate(search_queries, start=1):
    label = "原始问题" if i == 1 else "改写问题"
    print(f"{label}：{search_query}")

# 16. 每个查询分别执行两路召回，再统一进行 RRF 融合
scores = {}
doc_map = {}

for search_query in search_queries:
    bm25_results = bm25_retriever.invoke(search_query)
    vector_results = vector_retriever.invoke(search_query)

    for result_list in [bm25_results, vector_results]:
        # 防止同一条召回列表内重复计算同一个片段
        seen_keys = set()

        for rank, doc in enumerate(result_list, start=1):
            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", -1),
                doc.metadata.get("chunk_index", -1),
                doc.page_content,
            )

            if key in seen_keys:
                continue

            seen_keys.add(key)
            doc_map[key] = doc

            # 每个查询权重相同，每个查询内两路各占一半
            weight = 0.5 / len(search_queries)

            scores[key] = (
                scores.get(key, 0)
                + weight / (60 + rank)
            )

# 17. 按融合得分排序
ranked_keys = sorted(
    scores,
    key=scores.get,
    reverse=True,
)

# 保留最多 6 个候选，方便下一步接入重排
hybrid_results = [
    doc_map[key]
    for key in ranked_keys[:6]
]

# 18. 暂时只展示前 2 条，但候选列表仍保留最多 6 条
print("\n===== 多查询混合召回结果（前 2 条） =====")

if not hybrid_results:
    print("没有召回片段。")

for i, doc in enumerate(hybrid_results[:2], start=1):
    print(f"\n--- 第 {i} 条 ---")
    print(doc.page_content)

    source = doc.metadata.get("source", "未知来源")
    page = doc.metadata.get("page")
    page_number = page + 1 if isinstance(page, int) else "未知"

    print(f"来源：{source}")
    print(f"PDF 页码：{page_number}")