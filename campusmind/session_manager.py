from config import (
    HISTORY_ROUNDS,
)

from copy import deepcopy


class ChatSession:
    """管理当前会话的完整问答轮次，历史暂存在内存中。"""

    def __init__(self, history_rounds: int = HISTORY_ROUNDS):
        """
        history_rounds：默认取最近几轮问答作为上下文。
        一轮 = 用户的一个问题 + 系统的一个回答。
        """
        if history_rounds < 1:
            raise ValueError("history_rounds 必须大于等于 1。")

        self.history_rounds = history_rounds

        # 每个 ChatSession 对象都有自己的历史记录
        self._turns: list[dict] = []

    def add_turn(
        self,
        question: str,
        answer: str,
        sources: list[dict] | None = None,
    ) -> None:
        """保存一轮成功的问答和参考来源。"""
        question = question.strip()
        answer = answer.strip()

        if not question:
            raise ValueError("问题不能为空。")

        if not answer:
            raise ValueError("回答不能为空。")

        turn = {
            "question": question,
            "answer": answer,
            # 复制数据，避免外部修改影响已保存的历史
            "sources": deepcopy(sources) if sources is not None else [],
        }

        self._turns.append(turn)

    def get_recent_history(
        self,
        rounds: int | None = None,
    ) -> list[dict]:
        """
        获取最近几轮完整问答，按时间从早到晚排列。
        供后面的查询处理模块使用。
        """
        if rounds is None:
            rounds = self.history_rounds

        if rounds < 1:
            raise ValueError("rounds 必须大于等于 1。")

        return deepcopy(self._turns[-rounds:])

    def get_messages(self) -> list[dict]:
        """
        返回界面展示用的全部消息。
        每轮问答转换为一条用户消息和一条助手消息。
        """
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
        """清空当前会话的所有历史记录。"""
        self._turns.clear()


def main() -> None:
    # 以下是测试数据，不是实际检索或模型生成的结果
    session = ChatSession(history_rounds=2) #修改这个参数，就能改变 get_recent_history() 默认读取的历史轮数

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

    # 新建对象，验证不同会话不会共用历史
    another_session = ChatSession()

    print("\n===== 会话隔离 =====")
    print(f"原会话消息数：{len(session.get_messages())}")
    print(f"新会话消息数：{len(another_session.get_messages())}")

    session.clear()

    print("\n===== 清空会话 =====")
    print(f"清空后消息数：{len(session.get_messages())}")


if __name__ == "__main__":
    main()