# 图文PDF文档加载
# 打开 PDF → 页面转图片 → Base64 编码 → 组合图片和问题 → 调用模型 → 显示回答
import pymupdf
from PIL import Image

image_pdf_path = r"D:\VsCodeProjects\CampusMind\data\campus_multimodal.pdf"
pdf_document = pymupdf.open(image_pdf_path)
page = pdf_document.load_page(0)

print("图文 PDF 总页数：", len(pdf_document))

# PDF 页面转换成图片
pix = page.get_pixmap(colorspace=pymupdf.csRGB, alpha=False)

img = Image.frombytes(
    "RGB",
    (pix.width, pix.height),
    pix.samples
)

# 图片保存下来
image_path = r"D:\VsCodeProjects\CampusMind\data\campus_page_1.png"

img.save(image_path)
pdf_document.close()

print("图片已保存到：", image_path)

# 图片转换成 Base64 字符串
import base64

with open(image_path, "rb") as image_file:
    image_bytes = image_file.read()

base64_image = base64.b64encode(image_bytes).decode("utf-8")

print("编码后的前60个字符：", base64_image[:60])

# 读取DeepSeek 密钥
import os
from dotenv import load_dotenv

load_dotenv(r"D:\VsCodeProjects\CampusMind\.env")

api_key = os.getenv("DEEPSEEK_API_KEY")

print("是否读取到密钥：", bool(api_key))

# 创建模型客户端
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-flash",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    temperature=0,
)

print("模型客户端已创建")

# 把问题和图片放进同一条消息
from langchain_core.messages import HumanMessage

query = "请看图中的座位统计，告诉我哪一层空闲座位最多，有多少个？"

message = HumanMessage(
    content=[
        {"type": "text", "text": query},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            },
        },
    ]
)

print("图文消息已准备好")

# 发送图文消息并打印模型回答
response = llm.invoke([message])

print("模型回答：")
print(response.content)