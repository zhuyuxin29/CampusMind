from config import (
    PROJECT_DIR,
    RERANK_URL,
    RERANK_MODEL,
    CONTEXT_TOP_N,
    MAX_CONTEXT_CHARS,
    CONNECT_TIMEOUT,
    REQUEST_TIMEOUT,
    RERANK_MAX_RETRIES,
    get_api_key,
)

import logging
import math
from copy import deepcopy
from pathlib import Path

import jieba
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from langchain_core.documents import Document

from campusmind.knowledge_base import (
    COLLECTION_NAME,
    load_parent_documents,
)
from campusmind.query_processing import (
    ClarificationNeeded,
    contextualize_question,
    normalize_queries,
    rewrite_queries,
    validate_question,
)
from campusmind.retrieval import (
    build_hybrid_retriever,
    retrieve_candidates,
    format_document_source,
)


# 最多选取三份不同父级资料；正文总字数上限，不包含提示词和历史。


def _post_rerank(*args, **kwargs):
    """仅对连接失败、限流和暂时性服务错误做有限重试。"""
    retry = Retry(total=RERANK_MAX_RETRIES, backoff_factor=0.5,
                  status_forcelist=(429, 502, 503, 504), allowed_methods={"POST"},
                  respect_retry_after_header=False, raise_on_status=False)
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return session.post(*args, **kwargs)


def rerank_documents(
    question: str,
    candidates: list[Document],
) -> list[Document]:
    """调用重排模型，对候选片段重新评分和排序。"""
    question = validate_question(question)

    if not candidates:
        return []

    for index, doc in enumerate(candidates):
        if not doc.page_content.strip():
            raise ValueError(f"第 {index + 1} 个候选片段为空。")

    api_key = get_api_key("SILICONFLOW_API_KEY")

    payload = {
        "model": RERANK_MODEL,
        # 使用完整问题，不使用某一条关键词改写
        "query": question,
        "documents": [
            doc.page_content for doc in candidates
        ],
        # 返回全部候选的分数，后面由 select_context 选取
        "top_n": len(candidates),
        "return_documents": False,
    }

    try:
        response = _post_rerank(
            RERANK_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(CONNECT_TIMEOUT, REQUEST_TIMEOUT),
        )
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"重排请求失败（{type(exc).__name__}），请检查网络后重试。"
        ) from exc

    if not response.ok:
        raise RuntimeError(
            f"重排接口返回 HTTP {response.status_code}，"
            "请检查接口配置、账户额度和模型权限。"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("重排接口返回的内容不是有效 JSON。") from exc

    return parse_rerank_results(data, candidates)


def parse_rerank_results(
    response: dict,
    candidates: list[Document],
) -> list[Document]:
    """按返回的 index 映射原始片段，保留来源和其他元数据。"""
    if not isinstance(response, dict):
        raise ValueError("重排响应必须是 JSON 对象。")

    results = response.get("results")

    if not isinstance(results, list):
        raise ValueError("重排响应缺少有效的 results 列表。")

    # 请求时要求返回全部候选，因此这里检查完整性
    if len(results) != len(candidates):
        raise ValueError(
            f"重排结果不完整：输入 {len(candidates)} 个片段，"
            f"返回 {len(results)} 个结果。"
        )

    ranked_docs = []
    seen_indices = set()

    for item in results:
        if not isinstance(item, dict):
            raise ValueError("重排结果中的单条记录格式不正确。")

        index = item.get("index")
        score = item.get("relevance_score")

        if type(index) is not int:
            raise ValueError("重排结果的 index 必须是整数。")

        if not 0 <= index < len(candidates):
            raise ValueError(f"重排结果的 index 超出范围：{index}")

        if index in seen_indices:
            raise ValueError(f"重排结果出现重复 index：{index}")

        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("重排结果缺少有效的相关性分数。")

        score = float(score)

        if not math.isfinite(score):
            raise ValueError("重排分数不能是 NaN 或无穷大。")

        seen_indices.add(index)

        # index 从 0 开始，对应提交给接口的候选位置
        original = candidates[index]
        metadata = original.metadata.copy()

        metadata.update(
            {
                "rerank_score": score,
                "rerank_input_index": index,
            }
        )

        # 新建 Document，不修改原来的候选片段
        ranked_docs.append(
            Document(
                page_content=original.page_content,
                metadata=metadata,
            )
        )

    # 明确按分数降序排序；同分时保留原候选顺序
    ranked_docs.sort(
        key=lambda doc: (
            -doc.metadata["rerank_score"],
            doc.metadata["rerank_input_index"],
        )
    )

    for rank, doc in enumerate(ranked_docs, start=1):
        doc.metadata["rerank_rank"] = rank

    return ranked_docs


def select_context(
    ranked_docs: list[Document],
    top_n: int = CONTEXT_TOP_N,
    *,
    max_chars: int = MAX_CONTEXT_CHARS,
    parent_map: dict[str, Document] | None = None,
    collection_name: str = COLLECTION_NAME,
) -> list[Document]:
    """按子片段排名扩展完整父级、按 parent_id 去重，并控制正文预算。

    父级来自当前集合的版本快照。保留父级原文和页码，不用子片段元数据
    覆盖父级来源；不能放入剩余预算的资料跳过，不截断条款或例外。
    """
    if type(top_n) is not int or top_n < 1:
        raise ValueError("top_n 必须大于等于 1。")
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars 必须是正整数。")
    if not ranked_docs:
        return []
    if parent_map is None:
        try:
            parent_map = load_parent_documents(collection_name)
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError("无法读取当前集合的父级快照，请检查知识库版本记录。") from exc

    selected = []
    seen = set()
    total_chars = 0
    for child in ranked_docs:
        parent_id = child.metadata.get("parent_id")
        if not isinstance(parent_id, str) or not parent_id or parent_id not in parent_map:
            raise ValueError("候选片段缺少有效父级关联，请检查 v2 集合与父级快照。")
        if parent_id in seen:
            continue
        parent = parent_map[parent_id]
        if parent.metadata.get("parent_id") != parent_id or not parent.page_content.strip():
            raise ValueError("父级资料的编号或正文无效。")
        for key in ("source", "section_id", "scholarship_type"):
            if child.metadata.get(key) != parent.metadata.get(key):
                raise ValueError(f"父子资料的 {key} 不一致，请检查索引版本。")
        seen.add(parent_id)
        size = len(parent.page_content)
        if size > max_chars and not selected:
            raise ValueError(f"最高排名父级资料有 {size} 字，超过 {max_chars} 字预算，请调大 max_chars。")
        if total_chars + size > max_chars:
            logging.getLogger(__name__).warning("跳过无法完整放入剩余字数预算的父级资料：%s", parent_id)
            continue
        metadata = deepcopy(parent.metadata)
        # 重排分数属于命中的子片段，作为父级优先级；来源字段仍属于父级。
        for key in ("rerank_score", "rerank_rank", "rerank_input_index", "rrf_score"):
            if key in child.metadata:
                metadata[key] = child.metadata[key]
        metadata.update({
            "matched_chunk_id": child.metadata.get("chunk_id"),
            "context_level": "parent",
            "context_chars": size,
        })
        selected.append(Document(page_content=parent.page_content, metadata=metadata))
        total_chars += size
        if len(selected) == top_n:
            break
    return selected


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    jieba.setLogLevel(logging.WARNING)

    # 启动时创建一次混合检索器，后续提问复用
    retriever = build_hybrid_retriever(COLLECTION_NAME)

    print("===== 混合召回＋重排测试 =====")
    print(f"集合：{COLLECTION_NAME}；父级最多 {CONTEXT_TOP_N} 份，正文预算 {MAX_CONTEXT_CHARS} 字")
    print("请输入完整问题，输入 exit 退出。")
    print("当前测试不保存多轮对话，也不生成最终回答。")

    while True:
        question = input("\n请输入问题：").strip()

        if question.lower() in {"exit", "quit"}:
            break

        try:
            # 2. 查询处理
            question = validate_question(question)

            standalone_question = contextualize_question(
                question,
                history=[],
            )

            rewrites = rewrite_queries(standalone_question)
            queries = normalize_queries(
                standalone_question,
                rewrites,
            )

            print("\n===== 实际检索查询 =====")
            print(f"完整问题：{standalone_question}")

            for index, query in enumerate(queries, start=1):
                print(f"{index}. {query}")

            # 3. 多查询混合召回
            candidates = retrieve_candidates(
                queries=queries,
                retriever=retriever,
            )

            if not candidates:
                print("没有召回候选片段。")
                continue

            # 4. 使用完整问题进行重排
            print(
                f"\n正在对 {len(candidates)} 个候选片段重排……",
                flush=True,
            )

            ranked_docs = rerank_documents(
                question=standalone_question,
                candidates=candidates,
            )

            # 5. 选择后续回答使用的资料
            context_docs = select_context(
                ranked_docs,
                top_n=CONTEXT_TOP_N,
            )

        except ClarificationNeeded as exc:
            print(f"需要补充信息：{exc}")
            continue
        except (ValueError, RuntimeError) as exc:
            print(f"处理未完成：{exc}")
            continue

        # 查看全部候选的排名变化和分数分布
        print("\n===== 全部候选的重排分数 =====")

        for doc in ranked_docs:
            original_rank = doc.metadata["rerank_input_index"] + 1

            print(
                f"原排名 {original_rank:2d}"
                f" → 重排 {doc.metadata['rerank_rank']:2d}"
                f" | 分数 {doc.metadata['rerank_score']:.4f}"
                f" | {format_document_source(doc)}"
            )

        print(
            f"\n===== 父级参考资料：{len(context_docs)} 份，共 {sum(len(d.page_content) for d in context_docs)} 字 ====="
        )

        for index, doc in enumerate(context_docs, start=1):
            print(f"\n--- 第 {index} 条 ---")
            print(f"重排分数：{doc.metadata['rerank_score']:.4f}")
            print(f"来源：{format_document_source(doc)}")
            print(f"父级 ID：{doc.metadata['parent_id']}")
            print(doc.page_content)


if __name__ == "__main__":
    main()
