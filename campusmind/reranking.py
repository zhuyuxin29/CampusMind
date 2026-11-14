from config import (
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

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from langchain_core.documents import Document

from campusmind.knowledge_base import (
    COLLECTION_NAME,
    load_parent_documents,
)
from campusmind.errors import ServiceError
from campusmind.query_processing import validate_question


def _post_rerank(*args, **kwargs):
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
    question = validate_question(question)

    if not candidates:
        return []

    for index, doc in enumerate(candidates):
        if not doc.page_content.strip():
            raise ValueError(f"第 {index + 1} 个候选片段为空。")

    api_key = get_api_key("SILICONFLOW_API_KEY")

    payload = {
        "model": RERANK_MODEL,

        "query": question,
        "documents": [
            doc.page_content for doc in candidates
        ],

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
        raise ServiceError(
            f"重排请求失败（{type(exc).__name__}），请检查网络后重试。"
        ) from exc

    if not response.ok:
        raise ServiceError(
            f"重排接口返回 HTTP {response.status_code}，"
            "请检查接口配置、账户额度和模型权限。"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise ServiceError("重排接口返回的内容不是有效 JSON。") from exc

    return parse_rerank_results(data, candidates)


def parse_rerank_results(
    response: dict,
    candidates: list[Document],
) -> list[Document]:
    if not isinstance(response, dict):
        raise ValueError("重排响应必须是 JSON 对象。")

    results = response.get("results")

    if not isinstance(results, list):
        raise ValueError("重排响应缺少有效的 results 列表。")


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


        original = candidates[index]
        metadata = original.metadata.copy()

        metadata.update(
            {
                "rerank_score": score,
                "rerank_input_index": index,
            }
        )


        ranked_docs.append(
            Document(
                page_content=original.page_content,
                metadata=metadata,
            )
        )


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
    stats: dict | None = None,
) -> list[Document]:
    if type(top_n) is not int or top_n < 1:
        raise ValueError("top_n 必须大于等于 1。")
    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars 必须是正整数。")
    if stats is not None and not isinstance(stats, dict):
        raise ValueError("stats 必须是字典。")
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
    skipped = 0
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
        if size > max_chars or total_chars + size > max_chars:
            skipped += 1
            logging.getLogger(__name__).warning(
                "跳过无法完整放入剩余字数预算的父级资料：%s（%s 字，剩余 %s 字）",
                parent_id, size, max_chars - total_chars,
            )
            continue
        metadata = deepcopy(parent.metadata)

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

    for document in selected:
        document.metadata["context_incomplete"] = skipped > 0
    if stats is not None:
        stats.update({
            "selected_count": len(selected),
            "skipped_count": skipped,
            "truncated": skipped > 0,
            "context_chars": total_chars,
        })
    return selected


def main() -> None:
    from campusmind.cli import run

    run("rerank")


if __name__ == "__main__":
    main()
