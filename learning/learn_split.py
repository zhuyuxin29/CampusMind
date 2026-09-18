# 普通PDF文档加载：读取并打印所有的页面（单页和多页PDF都可以用）
# 完成四件事：导入加载器 → 指定 PDF → 读取内容 → 查看结果。
from langchain_community.document_loaders import PyPDFLoader

# 替换为自己的PDF路径
file_path = r"D:\VsCodeProjects\CampusMind\data\campus_text.pdf"

# 创建加载器并按页读取PDF内容
loader = PyPDFLoader(file_path)
pages = loader.load()

# # 打印总页数
# print("总页数：", len(pages))

# # 依次打印每一页的正文和元数据
# # enumerate(pages, start=1)：依次取出每一页，并从 1 开始编号。
# for page_number, page in enumerate(pages, start=1):
#     print(f"\n--- 第 {page_number} 页 ---")

#     print("正文：")
#     print(page.page_content)

#     print("元数据：")
#     print(page.metadata)


# --------------------------------------------------------------------------
# # 1、文本切分（按固定的长度切分）
# from langchain_text_splitters import CharacterTextSplitter

# # 创建切分器
# text_splitter = CharacterTextSplitter(
#     separator="\n",       # 按换行位置切分
#     chunk_size=100,       # 每块的目标大小为100个字符
#     chunk_overlap=20,     # 相邻块的目标重叠长度为20个字符
#     length_function=len,
# )

# # 对已经加载的 PDF 文档进行切分
# chunks = text_splitter.split_documents(pages)

# # 查看切分结果
# print("切分后的块数：", len(chunks))

# for i, chunk in enumerate(chunks, start=1):
#     print(f"\n--- 第 {i} 块 ---")
#     print("字符数：", len(chunk.page_content))
#     print(chunk.page_content)
#     print("元数据：", chunk.metadata)


# 2、按递归字符切分器进行切分
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    # 默认分隔符：空行 → 换行 → 空格 → 单字符
)

chunks = text_splitter.split_documents(pages)

print("切分后的块数：", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- 第 {i} 块 ---")
    print("字符数：", len(chunk.page_content))
    print(chunk.page_content)


# # 3、按句进行切分
# import re
# from langchain_core.documents import Document


# def chunk_by_sentences(pages, max_chars=100, overlap_sentences=1):
#     if max_chars <= 0 or overlap_sentences < 0:
#         raise ValueError("块大小必须大于0，重叠句数不能小于0")

#     chunks = []

#     # 每页单独切分，保留该页的元数据
#     for page in pages:
#         sentences = [
#             sentence.strip()
#             for sentence in re.split(
#                 r"(?<=[。！？])|\n+", # 如果是英文就换成r"(?<=[.!?])\s+|\n+",
#                 page.page_content
#             )
#             if sentence.strip()
#         ]

#         current_sentences = []

#         for sentence in sentences:
#             candidate = "\n".join(current_sentences + [sentence])

#             # 加入当前句后超长，先保存已有内容
#             if current_sentences and len(candidate) > max_chars:
#                 chunks.append(
#                     Document(
#                         page_content="\n".join(current_sentences),
#                         metadata=page.metadata.copy()
#                     )
#                 )

#                 # 保留上一块末尾的若干句
#                 if overlap_sentences > 0:
#                     current_sentences = current_sentences[
#                         -overlap_sentences:
#                     ]
#                 else:
#                     current_sentences = []

#                 # 重叠内容太长时，减少重叠，给新句子腾出空间
#                 while (
#                     current_sentences
#                     and len("\n".join(current_sentences + [sentence]))
#                     > max_chars
#                 ):
#                     current_sentences.pop(0)

#             current_sentences.append(sentence)

#         # 保存这一页最后剩下的内容
#         if current_sentences:
#             chunks.append(
#                 Document(
#                     page_content="\n".join(current_sentences),
#                     metadata=page.metadata.copy()
#                 )
#             )

#     return chunks


# # 对已经加载好的 pages 进行切分
# chunks = chunk_by_sentences(
#     pages,
#     max_chars=100,
#     overlap_sentences=1
# )

# # 查看结果
# print("切分后的块数：", len(chunks))

# for i, chunk in enumerate(chunks, start=1):
#     print(f"\n--- 第 {i} 块 ---")
#     print("字符数：", len(chunk.page_content))
#     print(chunk.page_content)
#     print("元数据：", chunk.metadata)