"""统一问答服务。界面调用 ask(question, session)，不要再次保存本轮问答。

status: answered / insufficient / needs_clarification / invalid_input / error。
timings_ms 的单位是毫秒，未执行的阶段为 None；首次初始化耗时单独记录。
服务复用检索资源，ChatSession 必须由界面为每个用户单独创建。
同一会话的请求应串行提交。知识库重建后重启服务以重新加载缓存。

并发：检索资源（HybridRetriever 与父级快照）在首次请求时经 _resource_lock
惰性初始化，之后只读、不再变更，因此不同会话的 ask() 可并发调用；
同一 ChatSession 仍需串行（历史为可变状态，未加锁）。
"""

from config import (
    HISTORY_ROUNDS,
)

import json
from copy import deepcopy
from functools import lru_cache
from threading import Lock
from time import perf_counter

from langchain_core.documents import Document

from campusmind.generation import (
    build_messages, format_context, generate_answer,
    make_insufficient_answer, validate_citations,
)
from campusmind.knowledge_base import (
    COLLECTION_NAME, SPLIT_VERSION, load_parent_documents, open_vector_store,
)
from campusmind.query_processing import (
    ClarificationNeeded, contextualize_question, normalize_queries,
    rewrite_queries, validate_question,
)
from campusmind.reranking import rerank_documents, select_context
from campusmind.retrieval import (
    CANDIDATE_K, RECALL_K, HybridRetriever, build_bm25_retriever,
    build_vector_retriever, retrieve_candidates,
)
from campusmind.session_manager import ChatSession


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


class QAService:
    """延迟加载并复用知识库及 BM25；不在服务中保存用户聊天历史。

    线程安全：_get_resources() 用锁保证惰性初始化只执行一次；初始化完成后
    共享的检索资源只读，因此多个会话可并发调用 ask()。ChatSession 属于单个
    用户的可变状态，仍应由界面保证同一会话内串行提交。
    """

    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = collection_name
        self._resources = None
        self._resource_lock = Lock()

    def _get_resources(self):
        with self._resource_lock:
            if self._resources is not None:
                return self._resources
            # 先确认当前集合的父级快照已 ready，再打开对应向量库。
            parents = load_parent_documents(self.collection_name)
            store = open_vector_store(self.collection_name)
            stored = store.get(include=["documents", "metadatas"])
            ids = stored.get("ids") or []
            texts = stored.get("documents") or []
            metadatas = stored.get("metadatas") or []
            if not ids or not (len(ids) == len(texts) == len(metadatas)):
                raise ValueError("知识库为空或记录不完整，请检查入库结果。")
            documents = []
            for text, metadata in zip(texts, metadatas):
                if not isinstance(text, str) or not text.strip() or not isinstance(metadata, dict):
                    raise ValueError("知识库包含无效文字或元数据。")
                if metadata.get("split_version") != SPLIT_VERSION or metadata.get("parent_id") not in parents:
                    raise ValueError("子片段与当前集合的父级快照不匹配。")
                documents.append(Document(page_content=text, metadata=metadata))
            retriever = HybridRetriever(
                bm25_retriever=build_bm25_retriever(documents),
                vector_retriever=build_vector_retriever(store, k=min(RECALL_K, len(documents))),
                top_k=CANDIDATE_K,
            )
            # 只有全部初始化成功才缓存；失败后下次调用允许重试。
            # 赋值后 _resources 不再修改，后续并发只读访问是安全的。
            self._resources = (retriever, parents)
            return self._resources

    def ask(self, question: str, session: ChatSession) -> dict:
        """完成一轮问答并返回可直接展示、可 JSON 序列化的结果。

        仅 answered 和 insufficient 保存到会话；澄清、无效输入及故障不保存。
        引用校验失败不返回未经核验的回答正文，也不返回错误来源。
        """
        started = perf_counter()
        stage = "validate"
        result = {
            "question": question if isinstance(question, str) else "",
            "standalone_question": "", "queries": [],
            "answer": "", "sources": [], "status": "error",
            "message": "", "error_stage": None, "saved": False,
            "collection_name": self.collection_name,
            "candidate_count": 0, "context_count": 0,
            "timings_ms": {name: None for name in STAGES},
        }

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
            # Streamlit 热更新后旧会话仍在 session_state 中，但类对象身份可能变化。
            # 校验服务实际依赖的接口，保留原会话及历史，不依赖 isinstance。
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
            if candidates:
                ranked = run("rerank", rerank_documents, standalone, candidates)
                docs = run("select_context", select_context, ranked, parent_map=parents,
                           collection_name=self.collection_name)
                result["context_count"] = len(docs)

                def prepare():
                    context, mapping = format_context(docs)
                    return build_messages(standalone, context, history), mapping

                messages, source_map = run("build_messages", prepare)
                if source_map:
                    answer = run("generate", generate_answer, messages)
                    if answer != make_insufficient_answer():
                        def check_answer():
                            check = validate_citations(answer, source_map)
                            if not check["valid"] or not check["has_citations"]:
                                raise ValueError("引用校验失败。")
                            return [deepcopy(source_map[i]) for i in check["cited_ids"]]

                        sources = run("validate_citations", check_answer)
                        status = "answered"
            run("save_session", session.add_turn, question, answer, sources)
            result.update(answer=answer, sources=sources, status=status, saved=True)
        except ClarificationNeeded as exc:
            result.update(status="needs_clarification", message=str(exc), answer=str(exc))
        except Exception as exc:
            # 不把 SDK 异常正文（可能带请求数据）直接暴露给界面。
            if stage == "validate" and isinstance(exc, ValueError):
                result.update(status="invalid_input", message=str(exc), error_stage=stage)
            else:
                message = f"{STAGE_LABELS[stage]}未完成，请检查配置或稍后重试。"
                if stage == "initialize":
                    message = "知识库尚未就绪、内容为空或配置不匹配，请先完成资料入库并检查服务配置。"
                elif stage in {"contextualize", "rewrite", "retrieve", "rerank", "generate"}:
                    message = f"{STAGE_LABELS[stage]}服务暂时不可用，请检查网络、密钥及账户额度后重试。"
                if stage == "validate_citations":
                    message = "回答引用校验未通过，请重试。"
                result.update(status="error", message=message, error_stage=stage)
        finally:
            result["timings_ms"]["total"] = round((perf_counter() - started) * 1000, 3)
        return result


@lru_cache(maxsize=1)
def get_qa_service() -> QAService:
    """应用内复用一个服务；导入和创建服务时都不调用模型或加载数据库。"""
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
        print("阶段耗时（毫秒，null 表示未执行）：")
        print(json.dumps(result["timings_ms"], ensure_ascii=False))


if __name__ == "__main__":
    main()
