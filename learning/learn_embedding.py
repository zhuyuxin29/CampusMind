import os
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

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

print("PDF 总页数：", len(pages))

# 4. 切分文本
# 先保留你原来的参数，方便对照学习
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
)

chunks = text_splitter.split_documents(pages)

# 去掉没有正文的切片
chunks = [
    chunk for chunk in chunks
    if chunk.page_content.strip()
]

if not chunks:
    raise ValueError("没有提取到文字，请检查 PDF 是否为扫描件")

print("切分后的块数：", len(chunks))

# 5. 创建嵌入模型客户端
embeddings_model = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
    check_embedding_ctx_length=False,
    chunk_size=16,  # 嵌入请求的批次大小，不是文本切片长度
)

# 6. 创建或打开本地 Chroma 数据库
vector_store = Chroma(
    collection_name="campus_bge_m3_v1",
    embedding_function=embeddings_model,
    persist_directory=str(db_path),
)

# 7. 为每个切片生成固定 ID
# 同一路径、相同位置和相同正文，会生成相同 ID
chunk_ids = []

for i, chunk in enumerate(chunks):
    # Chroma 元数据保留字符串、数字和布尔值
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

# 8. 分批检查并写入，每批最多 16 个切片
batch_size = 16
added_count = 0
skipped_count = 0

for start in range(0, len(chunks), batch_size):
    batch_chunks = chunks[start:start + batch_size]
    batch_ids = chunk_ids[start:start + batch_size]

    # 查询这一批有哪些片段已经存在
    existing = vector_store.get(
        ids=batch_ids,
        include=[],
    )
    existing_ids = set(existing["ids"])

    new_chunks = []
    new_ids = []

    for chunk, chunk_id in zip(batch_chunks, batch_ids):
        if chunk_id in existing_ids:
            skipped_count += 1
        else:
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    # 只对尚未存入的片段生成向量并保存
    if new_chunks:
        vector_store.add_documents(
            documents=new_chunks,
            ids=new_ids,
        )
        added_count += len(new_chunks)

    processed = min(start + batch_size, len(chunks))
    print(f"处理进度：{processed}/{len(chunks)}")

# 9. 查看导入结果
print("\n--- 导入完成 ---")
print("本次新增片段：", added_count)
print("跳过已有片段：", skipped_count)
print("数据库保存位置：", db_path)

# 从数据库读回一条记录，检查正文、元数据和向量
sample = vector_store.get(
    ids=[chunk_ids[0]],
    include=["documents", "metadatas", "embeddings"],
)

print("\n--- 数据库中的第一条片段 ---")
print("正文：", sample["documents"][0])
print("元数据：", sample["metadatas"][0])
print("向量维度：", len(sample["embeddings"][0]))