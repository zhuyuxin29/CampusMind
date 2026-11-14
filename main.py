from config import HISTORY_ROUNDS
from campusmind.qa_service import ask
from campusmind.session_manager import ChatSession


def print_result(result: dict) -> None:
    status = result.get("status", "error")
    labels = {
        "answered": "回答完成",
        "insufficient": "资料不足",
        "needs_clarification": "需要补充信息",
        "invalid_input": "输入有误",
        "error": "处理未完成",
    }
    print(f"\n【{labels.get(status, '处理未完成')}】")
    print(result.get("answer") or result.get("message") or "暂时无法完成回答，请重试。")
    if status == "needs_clarification":
        print("请将具体类别或事项补充到完整问题中，再次提交。")

    sources = result.get("sources", [])
    if sources:
        print("\n参考来源：")
    for source in sources:
        start = source.get("page_start") or source.get("page_number")
        end = source.get("page_end") or start
        pages = "页码未标注" if start is None else f"PDF 第 {start} 页" if start == end else f"PDF 第 {start}–{end} 页"
        title = source.get("section_title") or "未标注章节"
        print(f"[{source['citation_id']}] {source.get('file_name', '未知来源')}｜{title}｜{pages}")

    if result.get("context_incomplete"):
        print("\n提示：本轮有部分资料因上下文长度预算未纳入，回答可能不完整。")

    elapsed = result.get("timings_ms", {}).get("total")
    if isinstance(elapsed, (int, float)):
        print(f"\n本轮耗时：{elapsed / 1000:.2f} 秒")


def main() -> None:
    session = ChatSession(history_rounds=HISTORY_ROUNDS)
    print("===== CampusMind 校园问答 =====")
    print("输入问题开始问答；clear 清空历史；exit 或 quit 退出。")
    print("首次提问需要加载知识库，请稍候。")
    try:
        while True:
            question = input("\n请输入问题：").strip()
            if not question:
                print("问题不能为空，请重新输入。")
                continue
            if question.lower() in {"exit", "quit"}:
                break
            if question.lower() == "clear":
                session.clear()
                print("当前会话已清空。")
                continue
            print("正在查找资料并整理回答……", flush=True)

            result = ask(question, session)
            print_result(result)
    except (EOFError, KeyboardInterrupt):
        print()
    print("已退出 CampusMind。")


if __name__ == "__main__":
    main()
