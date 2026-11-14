from config import ANSWER_ATTEMPTS, COLLECTION_NAME, HISTORY_ROUNDS, missing_api_keys

import json
import logging
from functools import lru_cache
from threading import Lock
from time import perf_counter

from campusmind.errors import ServiceError
from campusmind.generation import (
    CitationCheckFailed,
    build_citation_retry_messages,
    build_messages,
    format_context,
    generate_answer,
    is_insufficient_answer,
    make_insufficient_answer,
    select_cited_sources,
)
from campusmind.query_processing import (
    ClarificationNeeded, contextualize_question, normalize_queries,
    rewrite_queries, validate_question,
)
from campusmind.reranking import rerank_documents, select_context
from campusmind.retrieval import open_index, retrieve_candidates
from campusmind.session_manager import ChatSession


logger = logging.getLogger(__name__)


STAGES = (
    "validate", "history", "contextualize", "rewrite", "initialize",
    "retrieve", "rerank", "select_context", "build_messages", "generate",
    "validate_citations", "save_session",
)
STAGE_LABELS = dict(zip(STAGES, (
    "校验问题", "读取历史", "补全问题", "查询改写", "初始化知识库",
    "混合召回", "重排", "扩展父级资料", "组织提示词", "生成回答",
    "校验引用", "保存问答",
)))


def _failure_message(stage: str, exc: Exception) -> str:
    if isinstance(exc, ServiceError):
        message = str(exc).strip()
        if message:
            return message
    name = type(exc).__name__.lower()
    if isinstance(exc, TimeoutError) or "timeout" in name:
        return "服务响应超时，请稍后重试。"
    if isinstance(exc, ConnectionError) or "connection" in name:
        return "网络连接失败，请检查网络后重试。"
    if "ratelimit" in name or "rate_limit" in name or "quota" in name:
        return "接口请求过于频繁或额度不足，请稍后重试。"
    if "authentication" in name or "permissiondenied" in name or "unauthorized" in name:
        return "接口凭证无效或权限不足，请检查密钥配置。"
    if stage == "initialize":
        return "知识库尚未就绪、内容为空或配置不匹配，请先完成资料入库并检查服务配置。"
    if stage in {"contextualize", "rewrite", "retrieve", "rerank", "select_context",
                 "build_messages", "generate"}:
        return f"{STAGE_LABELS[stage]}服务暂时不可用，请检查网络、密钥及账户额度后重试。"
    return f"{STAGE_LABELS[stage]}未完成，请检查配置或稍后重试。"


class QAService:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = collection_name
        self._resources = None
        self._resource_lock = Lock()

    def _get_resources(self):
        with self._resource_lock:
            if self._resources is None:
                self._resources = open_index(self.collection_name)
            return self._resources

    def ask(self, question: str, session: ChatSession) -> dict:
        started = perf_counter()
        stage = "validate"
        result = {
            "question": question if isinstance(question, str) else "",
            "standalone_question": "", "queries": [],
            "answer": "", "sources": [], "status": "error",
            "message": "", "error_stage": None, "saved": False,
            "collection_name": self.collection_name,
            "candidate_count": 0, "context_count": 0,
            "context_incomplete": False, "insufficient_reason": None,
            "timings_ms": {name: None for name in STAGES},
        }


        missing = missing_api_keys()
        if missing:
            result.update(
                status="error", error_stage="config",
                message="未配置 " + "、".join(missing)
                        + "，请在项目根目录 .env 中填写后重启应用。",
            )
            result["timings_ms"]["total"] = round((perf_counter() - started) * 1000, 3)
            return result

        def run(name, function, *args, **kwargs):
            nonlocal stage
            stage = name
            tick = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                result["timings_ms"][name] = round((perf_counter() - tick) * 1000, 3)

        try:
            question = run("validate", validate_question, question)
            result["question"] = question


            if not all(callable(getattr(session, method, None))
                       for method in ("get_recent_history", "add_turn")):
                raise ValueError("会话不可用，请开启新对话后重试。")
            history = run("history", session.get_recent_history)
            standalone = run("contextualize", contextualize_question, question, history)
            result["standalone_question"] = standalone

            def make_queries():
                return normalize_queries(standalone, rewrite_queries(standalone))

            result["queries"] = run("rewrite", make_queries)
            retriever, parents = run("initialize", self._get_resources)
            candidates = run("retrieve", retrieve_candidates, result["queries"], retriever)
            result["candidate_count"] = len(candidates)
            answer = make_insufficient_answer()
            sources = []
            status = "insufficient"
            insufficient_reason = None
            if candidates:
                ranked = run("rerank", rerank_documents, standalone, candidates)
                selection_stats = {}
                docs = run(
                    "select_context", select_context, ranked, parent_map=parents,
                    collection_name=self.collection_name, stats=selection_stats,
                )
                result["context_count"] = len(docs)

                result["context_incomplete"] = bool(
                    selection_stats.get("truncated")
                    or any(doc.metadata.get("context_incomplete") for doc in docs)
                )

                def prepare():
                    context, mapping = format_context(docs)
                    return build_messages(standalone, context, history), mapping

                messages, source_map = run("build_messages", prepare)
                if source_map:
                    for attempt in range(ANSWER_ATTEMPTS):
                        prompt = messages if attempt == 0 else build_citation_retry_messages(messages)
                        answer = run("generate", generate_answer, prompt)
                        if is_insufficient_answer(answer):
                            answer = make_insufficient_answer()
                            insufficient_reason = "model_declined"
                            break
                        try:
                            sources = run(
                                "validate_citations",
                                select_cited_sources, answer, source_map,
                            )
                        except CitationCheckFailed:
                            sources = []
                            if attempt + 1 >= ANSWER_ATTEMPTS:
                                answer = make_insufficient_answer()
                                insufficient_reason = "citation_unverified"
                            continue
                        status = "answered"
                        break
            run("save_session", session.add_turn, question, answer, sources)
            result.update(
                answer=answer, sources=sources, status=status, saved=True,
                insufficient_reason=insufficient_reason,
            )
        except ClarificationNeeded as exc:
            result.update(status="needs_clarification", message=str(exc), answer=str(exc))
        except Exception as exc:
            logger.warning("问答未完成：阶段=%s 异常=%s", stage, type(exc).__name__)
            logger.debug("问答异常详情", exc_info=True)
            if stage == "validate" and isinstance(exc, ValueError):
                result.update(status="invalid_input", message=str(exc), error_stage=stage)
            else:
                result.update(status="error", message=_failure_message(stage, exc),
                              error_stage=stage)
        finally:
            result["timings_ms"]["total"] = round((perf_counter() - started) * 1000, 3)
        return result


@lru_cache(maxsize=1)
def get_qa_service() -> QAService:
    return QAService()


def ask(question: str, session: ChatSession) -> dict:
    return get_qa_service().ask(question, session)


def main() -> None:
    session = ChatSession(history_rounds=HISTORY_ROUNDS)
    print("CampusMind 问答服务：exit 退出，clear 清空历史。")
    while True:
        try:
            question = input("\n请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "clear":
            session.clear()
            print("历史已清空。")
            continue
        result = ask(question, session)
        print(result["answer"] or result["message"])
        for source in result["sources"]:
            start, end = source["page_start"], source["page_end"]
            pages = str(start) if start == end else f"{start}–{end}"
            print(f"[{source['citation_id']}] {source['file_name']}｜{source['section_title']}｜PDF 第 {pages} 页")
        print(f"状态：{result['status']}")
        if result.get("insufficient_reason"):
            print(f"资料不足原因：{result['insufficient_reason']}")
        print("阶段耗时（毫秒，null 表示未执行）：")
        print(json.dumps(result["timings_ms"], ensure_ascii=False))


if __name__ == "__main__":
    main()
