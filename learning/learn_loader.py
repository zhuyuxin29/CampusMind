# 普通PDF文档加载：读取并打印所有的页面（单页和多页PDF都可以用）
# 完成四件事：导入加载器 → 指定 PDF → 读取内容 → 查看结果。
from langchain_community.document_loaders import PyPDFLoader

# 替换为自己的PDF路径
file_path = r"D:\VsCodeProjects\CampusMind\data\campus_text.pdf"

# 创建加载器并按页读取PDF内容
loader = PyPDFLoader(file_path)
pages = loader.load()

# 打印总页数
print("总页数：", len(pages))

# 依次打印每一页的正文和元数据
# enumerate(pages, start=1)：依次取出每一页，并从 1 开始编号。
for page_number, page in enumerate(pages, start=1):
    print(f"\n--- 第 {page_number} 页 ---")

    print("正文：")
    print(page.page_content)

    print("元数据：")
    print(page.metadata)