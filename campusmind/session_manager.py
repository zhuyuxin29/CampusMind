from config import (
    HISTORY_ROUNDS,
)

from copy import deepcopy


class ChatSession:
    def __init__(self, history_rounds: int = HISTORY_ROUNDS):
        if history_rounds < 1:
            raise ValueError("history_rounds 必须大于等于 1。")

        self.history_rounds = history_rounds


        self._turns: list[dict] = []

    def add_turn(
        self,
        question: str,
        answer: str,
        sources: list[dict] | None = None,
    ) -> None:
        question = question.strip()
        answer = answer.strip()

        if not question:
            raise ValueError("问题不能为空。")

        if not answer:
            raise ValueError("回答不能为空。")

        turn = {
            "question": question,
            "answer": answer,

            "sources": deepcopy(sources) if sources is not None else [],
        }

        self._turns.append(turn)

    def get_recent_history(
        self,
        rounds: int | None = None,
    ) -> list[dict]:
        if rounds is None:
            rounds = self.history_rounds

        if rounds < 1:
            raise ValueError("rounds 必须大于等于 1。")

        return deepcopy(self._turns[-rounds:])

    def get_messages(self) -> list[dict]:
        messages = []

        for turn in self._turns:
            messages.append(
                {
                    "role": "user",
                    "content": turn["question"],
                }
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": turn["answer"],
                    "sources": deepcopy(turn["sources"]),
                }
            )

        return messages

    def clear(self) -> None:
        self._turns.clear()


def main() -> None:
    session = ChatSession(history_rounds=2)

    session.add_turn(
        question="如何申请缓考？",
        answer="这是第一轮测试回答。",
        sources=[
            {
                "file_name": "湖南信息学院学生手册.pdf",
                "page_number": 44,
            }
        ],
    )

    session.add_turn(
        question="需要准备什么材料？",
        answer="这是第二轮测试回答。",
    )

    session.add_turn(
        question="应该交给谁？",
        answer="这是第三轮测试回答。",
    )

    print("===== 最近两轮问答 =====")
    recent_history = session.get_recent_history()

    for index, turn in enumerate(recent_history, start=1):
        print(f"\n第 {index} 轮")
        print(f"问题：{turn['question']}")
        print(f"回答：{turn['answer']}")

    print("\n===== 界面消息 =====")
    messages = session.get_messages()
    print(f"消息数量：{len(messages)}")

    for message in messages:
        print(f"{message['role']}：{message['content']}")


    another_session = ChatSession()

    print("\n===== 会话隔离 =====")
    print(f"原会话消息数：{len(session.get_messages())}")
    print(f"新会话消息数：{len(another_session.get_messages())}")

    session.clear()

    print("\n===== 清空会话 =====")
    print(f"清空后消息数：{len(session.get_messages())}")


if __name__ == "__main__":
    main()
