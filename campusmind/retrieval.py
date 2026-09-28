from config import (
    RECALL_K,
    CANDIDATE_K,
    RRF_CONSTANT,
)

import logging
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
    make_chunk_id,
    open_vector_store,
)
from campusmind.query_processing import (
    ClarificationNeeded,
    contextualize_question,
    normalize_queries,
    rewrite_queries,
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


def build_hybrid_retriever(
    collection_name: str = COLLECTION_NAME,
    *,
    top_k: int = CANDIDATE_K,
) -> HybridRetriever:
    """打开集合、读取子片段并建立 BM25 + 向量混合检索器。

    供在线问答服务与各调试入口复用；校验知识库非空、记录完整且文字与元数据有效。
    """
    store = open_vector_store(collection_name)
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
        documents.append(Document(page_content=text, metadata=metadata))
    if not documents:
        raise ValueError("知识库中没有有效文字。")
    return HybridRetriever(
        bm25_retriever=build_bm25_retriever(documents),
        vector_retriever=build_vector_retriever(store, k=min(RECALL_K, len(documents))),
        top_k=top_k,
    )


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
    logging.basicConfig(level=logging.WARNING)
    jieba.setLogLevel(logging.WARNING)

    # 启动时建立一次混合检索器，后续提问复用。
    retriever = build_hybrid_retriever(COLLECTION_NAME)

    print(f"集合：{COLLECTION_NAME}；混合检索器已就绪")
    print("输入完整问题开始检索，输入 exit 退出。")
    print("当前是检索测试，不保存多轮对话，也不生成回答。")

    # 多次提问复用同一检索器，不重复建立 BM25
    while True:
        question = input("\n请输入问题：").strip()

        if question.lower() in {"exit", "quit"}:
            break

        try:
            # 3. 接入查询处理模块
            question = validate_question(question)

            # 本测试没有真实问答历史，传入空列表
            standalone_question = contextualize_question(
                question,
                history=[],
            )

            rewrites = rewrite_queries(standalone_question)
            queries = normalize_queries(
                standalone_question,
                rewrites,
            )

        except ClarificationNeeded as exc:
            print(f"需要补充信息：{exc}")
            continue
        except (ValueError, RuntimeError) as exc:
            print(f"查询处理未完成：{exc}")
            continue

        print("\n===== 实际检索查询 =====")
        for index, query in enumerate(queries, start=1):
            print(f"{index}. {query}")

        # 4. 多查询混合召回
        candidates = retrieve_candidates(
            queries=queries,
            retriever=retriever,
        )

        print(
            f"\n===== 混合召回结果：{len(candidates)} 个候选 ====="
        )

        for index, doc in enumerate(candidates, start=1):
            print(f"\n--- 第 {index} 条 ---")
            print(f"片段 ID：{doc.metadata['chunk_id']}")
            print(f"RRF 分数：{doc.metadata['rrf_score']:.6f}")
            print(f"来源：{format_document_source(doc)}")
            print(doc.page_content)


if __name__ == "__main__":
    main()
