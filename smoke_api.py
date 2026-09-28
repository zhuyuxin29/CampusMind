"""DeepSeek 接口冒烟测试脚本。

手动运行：.venv/Scripts/python.exe smoke_api.py
会实际调用 DeepSeek 接口，可能消耗账户额度；不是离线测试，也不覆盖完整 RAG 流程。
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def get_completion(prompt):
    # 读取与当前脚本同一文件夹中的 .env
    load_dotenv(Path(__file__).parent / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("没有读取到 DEEPSEEK_API_KEY，请检查 .env 文件。")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    response = client.chat.completions.create(
        model="deepseek-flash",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    return response.choices[0].message.content


def main():
    text = """
RAG通过先检索外部知识，再把检索到的信息交给大语言模型，
从而帮助模型生成更准确的答案。
"""

    prompt = f"""
请总结下面这段文字。

要求：
1. 使用通俗易懂的中文
2. 不超过30字
3. 只输出总结结果

文本：
{text}
"""

    print(get_completion(prompt))


if __name__ == "__main__":
    main()
