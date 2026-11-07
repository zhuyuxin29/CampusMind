from config import (
    RECALL_K,
    CANDIDATE_K,
    RRF_CONSTANT,
)

from collections import defaultdict

import jieba
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from campusmind.knowledge_base import (
    COLLECTION_NAME,
    SPLIT_VERSION,
    load_index_manifest,
    load_parent_documents,
    make_chunk_id,
    open_vector_store,
)
from campusmind.query_processing import (
    normalize_queries,
    validate_question,
)


# 每条查询在每一路最多召回 6 个片段

# 所有结果融合后，最多保留 10 个候选，交给后续重排

# RRF 排名融合的平滑参数，不是返回数量


def tokenize_chinese(text: str) -> list[str]:
    """使用 jieba 分词，去除纯空白和纯标点。"""
    tokens = []

    for word in jieba.lcut(text.lower()):
        word = word.strip()

        if word and any(char.isalnum() for char in word):
            tokens.append(word)

    return tokens


def build_bm25_retriever(
    documents: list[Document],
    k: int = RECALL_K,
) -> BM25Retriever:
    """根据知识库文字创建 BM25 检索器。"""
    if not documents:
        raise ValueError("知识库没有文档，无法建立 BM25 检索器。")

    if k < 1:
        raise ValueError("k 必须大于等于 1。")

    if not any(tokenize_chinese(doc.page_content) for doc in documents):
        raise ValueError("知识库文字没有可用于 BM25 的有效词语。")

    return BM25Retriever.from_documents(
        documents=documents,
        preprocess_func=tokenize_chinese,
        k=min(k, len(documents)),
    )


def build_vector_retriever(
    store: Chroma,
    k: int = RECALL_K,
) -> BaseRetriever:
    """把已有 Chroma 集合转换为向量检索器。"""
    if k < 1:
        raise ValueError("k 必须大于等于 1。")

    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


def _get_chunk_id(doc: Document) -> str:
    """使用与入库一致的算法识别同一个片段。"""
    if doc.metadata.get("split_version") == SPLIT_VERSION:
        return make_chunk_id(doc, 0)
    if "chunk_index" not in doc.metadata:
        raise ValueError("片段缺少 chunk_index，请检查入库数据。")

    return make_chunk_id(
        doc,
        int(doc.metadata["chunk_index"]),
    )


def fuse_rankings(
    result_lists: list[list[Document]],
    top_k: int = CANDIDATE_K,
    rrf_constant: int = RRF_CONSTANT,
) -> list[Document]:
    """按片段 ID 去重，通过 RRF 融合多个排名列表。"""
    if top_k < 1:
        raise ValueError("top_k 必须大于等于 1。")

    if rrf_constant < 1:
        raise ValueError("rrf_constant 必须大于等于 1。")

    if not result_lists:
        return []

    scores = defaultdict(float)
    documents_by_id = {}

    # 所有查询、所有召回路使用相同权重
    list_weight = 1.0 / len(result_lists)

    for documents in result_lists:
        # 同一列表中重复的片段只计分一次
        seen_in_list = set()

        for rank, doc in enumerate(documents, start=1):
            chunk_id = _get_chunk_id(doc)

            if chunk_id in seen_in_list:
                continue

            seen_in_list.add(chunk_id)
            documents_by_id.setdefault(chunk_id, doc)

            scores[chunk_id] += (
                list_weight / (rrf_constant + rank)
            )

    # 相同分数时按 ID 排序，保证结果顺序稳定
    ranked_ids = sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], chunk_id),
    )[:top_k]

    results = []

    for rank, chunk_id in enumerate(ranked_ids, start=1):
        original = documents_by_id[chunk_id]
        metadata = original.metadata.copy()

        metadata.update(
            {
                "chunk_id": chunk_id,
                "rrf_score": scores[chunk_id],
                "retrieval_rank": rank,
            }
        )

        # 返回新对象，不修改 BM25 中缓存的原始文档
        results.append(
            Document(
                page_content=original.page_content,
                metadata=metadata,
            )
        )

    return results


class HybridRetriever(BaseRetriever):
    """支持 LangChain invoke() 接口的混合检索器。"""

    bm25_retriever: BM25Retriever
    vector_retriever: BaseRetriever
    top_k: int = CANDIDATE_K

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """invoke(query) 调用时，执行单查询两路召回和融合。"""
        query = validate_question(query)

        if not tokenize_chinese(query):
            raise ValueError("问题中没有可检索的文字。")

        bm25_docs = self.bm25_retriever.invoke(
            query,
            config={"callbacks": run_manager.get_child()},
        )
        vector_docs = self.vector_retriever.invoke(
            query,
            config={"callbacks": run_manager.get_child()},
        )

        return fuse_rankings(
            [bm25_docs, vector_docs],
            top_k=self.top_k,
        )


def retrieve_candidates(
    queries: list[str],
    retriever: HybridRetriever,
) -> list[Document]:
    """执行多查询、多路召回，将全部原始排名列表统一融合。"""
    if not queries:
        raise ValueError("查询列表不能为空。")

    # 第一条应是补全后的完整问题，限制最多三个查询
    queries = normalize_queries(
        original=queries[0],
        rewrites=queries[1:],
    )

    result_lists = []

    for query in queries:
        if not tokenize_chinese(query):
            raise ValueError("查询中没有可检索的文字。")

        result_lists.append(
            retriever.bm25_retriever.invoke(query)
        )
        result_lists.append(
            retriever.vector_retriever.invoke(query)
        )

    # 先汇集各路原始结果，再统一融合，避免提前截断
    return fuse_rankings(
        result_lists,
        top_k=retriever.top_k,
    )


def open_index(
    collection_name: str = COLLECTION_NAME,
    *,
    top_k: int = CANDIDATE_K,
) -> tuple[HybridRetriever, dict[str, Document]]:
    """打开集合并校验索引契约，返回混合检索器和父级快照。

    在线问答服务与各调试入口共用这一入口：集合非空、记录完整，且子片段
    版本与父级快照必须一致，否则拒绝启动，避免静默使用错配的资料。
    """
    record = load_index_manifest(collection_name)
    parents = load_parent_documents(collection_name)
    store = open_vector_store(collection_name)
    stored = store.get(include=["documents", "metadatas"])
    ids = stored.get("ids") or []
    texts = stored.get("documents") or []
    metadatas = stored.get("metadatas") or []
    if not ids or not (len(ids) == len(texts) == len(metadatas)):
        raise ValueError("知识库为空或记录不完整，请检查入库结果。")
    if len(ids) != record["chunk_count"]:
        raise ValueError(
            f"知识库片段数与版本记录不一致（实际 {len(ids)}，记录 {record['chunk_count']}），"
            "请重新入库。"
        )
    documents = []
    for text, metadata in zip(texts, metadatas):
        if not isinstance(text, str) or not text.strip() or not isinstance(metadata, dict):
            raise ValueError("知识库包含无效文字或元数据。")
        if metadata.get("split_version") != SPLIT_VERSION or metadata.get("parent_id") not in parents:
            raise ValueError("子片段与当前集合的父级快照不匹配。")
        documents.append(Document(page_content=text, metadata=metadata))
    if not documents:
        raise ValueError("知识库中没有有效文字。")
    retriever = HybridRetriever(
        bm25_retriever=build_bm25_retriever(documents),
        vector_retriever=build_vector_retriever(store, k=min(RECALL_K, len(documents))),
        top_k=top_k,
    )
    return retriever, parents


def build_hybrid_retriever(
    collection_name: str = COLLECTION_NAME,
    *,
    top_k: int = CANDIDATE_K,
) -> HybridRetriever:
    """调试入口：只返回混合检索器，启动约束与在线服务完全一致。"""
    retriever, _ = open_index(collection_name, top_k=top_k)
    return retriever


def format_document_source(doc: Document) -> str:
    """展示章节及资料实际覆盖的 PDF 起止页，兼容旧单页元数据。"""
    metadata = doc.metadata
    start = metadata.get("page_start", metadata.get("page_number", "?"))
    end = metadata.get("page_end", start)
    pages = str(start) if start == end else f"{start}–{end}"
    return (
        f"{metadata.get('file_name', '未知')}｜"
        f"{metadata.get('section_title') or '未标注章节'}｜PDF 第 {pages} 页"
    )


def main() -> None:
    """调试入口：见 campusmind.cli（--stage retrieve）。"""
    from campusmind.cli import run

    run("retrieve")


if __name__ == "__main__":
    main()
