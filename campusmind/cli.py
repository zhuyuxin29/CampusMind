"""统一调试 CLI：按阶段查看查询处理、召回、重排与回答。

用法（项目根目录，先完成资料入库）：
  python -m campusmind.cli --stage query       # 查询处理：补全 + 改写
  python -m campusmind.cli --stage retrieve    # 混合召回
  python -m campusmind.cli --stage rerank      # 重排 + 父级扩展
  python -m campusmind.cli --stage answer      # 完整问答（调用模型，保存会话）

各阶段共享同一个混合检索器；answer 阶段直接复用 QAService，与网页/终端
入口保持相同的引用重试与降级行为。原先分散在 query_processing/retrieval/
reranking/generation 各自 main() 的调试逻辑集中在此，各模块的 main() 仍可
运行并委托到这里。
"""

import argparse
import json
import logging

import jieba

from config import CONTEXT_TOP_N, HISTORY_ROUNDS
from campusmind.qa_service import QAService
from campusmind.knowledge_base import COLLECTION_NAME
from campusmind.query_processing import (
    ClarificationNeeded,
    contextualize_question,
    normalize_queries,
    rewrite_queries,
    validate_question,
)
from campusmind.reranking import rerank_documents, select_context
from campusmind.retrieval import (
    build_hybrid_retriever,
    format_document_source,
    retrieve_candidates,
)
from campusmind.session_manager import ChatSession


STAGES = ("query", "retrieve", "rerank", "answer")


def _print_queries(queries: list[str]) -> None:
    print("\n===== 实际检索查询 =====")
    for index, query in enumerate(queries, start=1):
        print(f"{index}. {query}")


def _print_candidates(candidates) -> None:
    print(f"\n===== 混合召回结果：{len(candidates)} 个候选 =====")
    for index, doc in enumerate(candidates, start=1):
        print(f"\n--- 第 {index} 条 ---")
        print(f"片段 ID：{doc.metadata['chunk_id']}")
        print(f"RRF 分数：{doc.metadata['rrf_score']:.6f}")
        print(f"来源：{format_document_source(doc)}")
        print(doc.page_content)


def _print_ranked(ranked_docs) -> None:
    print("\n===== 全部候选的重排分数 =====")
    for doc in ranked_docs:
        original_rank = doc.metadata["rerank_input_index"] + 1
        print(
            f"原排名 {original_rank:2d}"
            f" → 重排 {doc.metadata['rerank_rank']:2d}"
            f" | 分数 {doc.metadata['rerank_score']:.4f}"
            f" | {format_document_source(doc)}"
        )


def _print_context(context_docs) -> None:
    print(
        f"\n===== 父级参考资料：{len(context_docs)} 份，"
        f"共 {sum(len(d.page_content) for d in context_docs)} 字 ====="
    )
    for index, doc in enumerate(context_docs, start=1):
        print(f"\n--- 第 {index} 条 ---")
        print(f"重排分数：{doc.metadata['rerank_score']:.4f}")
        print(f"来源：{format_document_source(doc)}")
        print(f"父级 ID：{doc.metadata['parent_id']}")
        print(doc.page_content)


def _print_answer(result: dict) -> None:
    """打印完整问答结果；answer 阶段与网页入口使用同一份结果结构。"""
    print(f"\n状态：{result.get('status', 'error')}")
    print(result.get("answer") or result.get("message") or "本轮没有返回内容。")
    if result.get("insufficient_reason"):
        print(f"资料不足原因：{result['insufficient_reason']}")
    if result.get("context_incomplete"):
        print("提示：本轮有部分资料因上下文长度预算未纳入，回答可能不完整。")
    sources = result.get("sources") or []
    if sources:
        print("\n参考来源（PDF 页码）：")
    for source in sources:
        start, end = source["page_start"], source["page_end"]
        pages = "未知" if start is None else str(start) if end == start else f"{start}–{end}"
        print(f"[{source['citation_id']}] {source['file_name']}，第 {pages} 页")
    timings = result.get("timings_ms") or {}
    if timings:
        print("\n阶段耗时（毫秒）：")
        print(json.dumps(timings, ensure_ascii=False))


def _resolve(question: str, history: list[dict]):
    """validate → contextualize → rewrite → normalize，返回完整问题与检索查询。"""
    question = validate_question(question)
    standalone = contextualize_question(question, history)
    rewrites = rewrite_queries(standalone)
    return standalone, normalize_queries(standalone, rewrites)


def run(stage: str) -> None:
    """进入指定阶段的交互式调试循环。"""
    logging.basicConfig(level=logging.WARNING)
    jieba.setLogLevel(logging.WARNING)

    if stage not in STAGES:
        raise ValueError(f"未知阶段：{stage}，可选 {'/'.join(STAGES)}")

    need_index = stage in {"retrieve", "rerank"}
    retriever = build_hybrid_retriever(COLLECTION_NAME) if need_index else None
    # answer 阶段复用统一问答服务，保留引用重试和失败降级行为。
    service = QAService(COLLECTION_NAME) if stage == "answer" else None
    session = ChatSession(history_rounds=HISTORY_ROUNDS)

    print(f"===== CampusMind 调试 · 阶段：{stage} =====")
    print(f"集合：{COLLECTION_NAME}")
    print("输入完整问题开始，exit 退出；answer 阶段可用 clear 清空会话。")

    while True:
        try:
            question = input("\n请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            # 与 main.py 一致：Ctrl+C / 关闭输入时正常退出，不打印堆栈。
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if stage == "answer":
            if question.lower() == "clear":
                session.clear()
                print("会话已清空。")
                continue
            _print_answer(service.ask(question, session))
            continue

        try:
            standalone, queries = _resolve(question, [])

            if stage == "query":
                print(f"\n完整问题：{standalone}")
                _print_queries(queries)
                continue

            candidates = retrieve_candidates(queries=queries, retriever=retriever)
            if stage == "retrieve":
                _print_candidates(candidates)
                continue
            if not candidates:
                print("没有召回候选片段。")
                continue

            print(f"\n正在对 {len(candidates)} 个候选片段重排……", flush=True)
            ranked_docs = rerank_documents(question=standalone, candidates=candidates)
            context_docs = select_context(ranked_docs, top_n=CONTEXT_TOP_N)
            if stage == "rerank":
                _print_ranked(ranked_docs)
                _print_context(context_docs)
                continue
        except ClarificationNeeded as exc:
            print(f"需要补充信息：{exc}")
        except (ValueError, RuntimeError) as exc:
            print(f"处理未完成：{exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CampusMind 分阶段调试 CLI。")
    parser.add_argument(
        "--stage", choices=STAGES, default="answer",
        help="调试阶段（默认 answer）",
    )
    args = parser.parse_args()
    run(args.stage)


if __name__ == "__main__":
    main()
