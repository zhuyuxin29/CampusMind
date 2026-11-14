from config import (
    PROJECT_DIR,
    PDF_PATH,
    PROCESSED_DIR,
    STRUCTURED_DIR,
    OCR_DIR,
    CHROMA_DIR,
    INDEX_ROOT,
    LEGACY_COLLECTION_NAME,
    COLLECTION_NAME,
    SPLIT_VERSION,
    EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
    BATCH_SIZE,
    REQUEST_TIMEOUT,
    EMBEDDING_MAX_RETRIES,
    get_api_key,
)

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def create_embeddings() -> OpenAIEmbeddings:
    api_key = get_api_key("SILICONFLOW_API_KEY")
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
        base_url=EMBEDDING_BASE_URL,
        check_embedding_ctx_length=False,
        chunk_size=BATCH_SIZE,
        request_timeout=REQUEST_TIMEOUT,
        max_retries=EMBEDDING_MAX_RETRIES,
    )


def open_vector_store(collection_name: str) -> Chroma:
    _validate_collection_name(collection_name)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=create_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def make_chunk_id(doc: Document, index: int) -> str:
    if doc.metadata.get("split_version") == SPLIT_VERSION:
        chunk_id = doc.metadata.get("chunk_id")
        if not isinstance(chunk_id, str) or not re.fullmatch(r"[0-9a-f]{64}", chunk_id):
            raise ValueError("新版片段缺少有效 chunk_id，请重新生成结构化结果。")
        return chunk_id
    info = {
        "source": doc.metadata["source"],
        "page_number": doc.metadata["page_number"],
        "chunk_index": doc.metadata.get("chunk_index", index),
        "page_content": doc.page_content,
    }
    text = json.dumps(info, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            hasher.update(block)
    return hasher.hexdigest()


def _read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary_path = Path(file.name)
        try:
            file.write(data)
        except BaseException:
            file.close()
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _save_json(path: Path, data: dict) -> None:
    _atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))


def _validate_collection_name(name: str) -> None:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{1,61}[A-Za-z0-9]", name):
        raise ValueError("集合名需为 3–63 位字母、数字、下划线或短横线，首尾为字母或数字。")


def _decode_documents(data: bytes, label: str) -> list[Document]:
    records = json.loads(data.decode("utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} 必须是非空列表。")
    documents = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{label} 中有无效记录。")
        text, metadata = record.get("page_content"), record.get("metadata")
        if not isinstance(text, str) or not text.strip() or not isinstance(metadata, dict):
            raise ValueError(f"{label} 中有空文字或无效元数据。")
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def _make_parent_map(parents: list[Document]) -> dict[str, Document]:
    parent_map = {}
    for parent in parents:
        parent_id = parent.metadata.get("parent_id")
        if not isinstance(parent_id, str) or not re.fullmatch(r"[0-9a-f]{64}", parent_id):
            raise ValueError("父级资料缺少有效 parent_id。")
        if parent_id in parent_map:
            raise ValueError("父级资料出现重复 parent_id。")
        parent_map[parent_id] = parent
    return parent_map


def _validate_relationships(chunks: list[Document], parent_map: dict[str, Document]) -> None:
    seen_ids = set()
    referenced_parents = set()
    for i, child in enumerate(chunks):
        m = child.metadata
        if m.get("split_version") != SPLIT_VERSION:
            raise ValueError("发现非第二版子片段，不能混合入库。")
        if any(type(value) not in (str, int, float, bool) for value in m.values()):
            raise ValueError("子片段元数据必须是标量，不能把父级的 page_spans 列表写入 Chroma。")
        if any(isinstance(value, float) and not math.isfinite(value) for value in m.values()):
            raise ValueError("子片段元数据中有非有限数值。")
        parent_id = m.get("parent_id")
        if parent_id not in parent_map:
            raise ValueError(f"子片段 {i} 找不到对应父级。")
        parent = parent_map[parent_id]
        referenced_parents.add(parent_id)
        for key in ("source", "section_id", "section_title", "scholarship_type"):
            if m.get(key) != parent.metadata.get(key):
                raise ValueError(f"子片段与父级的 {key} 不一致。")
        start, end, prefix = (m.get(key) for key in ("parent_start_index", "parent_end_index", "section_prefix_length"))
        if not all(type(value) is int for value in (start, end, prefix)):
            raise ValueError("父子片段的字符位置格式错误。")
        if not (0 <= start < end <= len(parent.page_content) and 0 <= prefix < len(child.page_content)):
            raise ValueError("父子片段的字符位置超出范围。")
        if child.page_content[prefix:] != parent.page_content[start:end]:
            raise ValueError("子片段正文与父级对应位置不一致。")
        pages = [span["page_number"] for span in parent.metadata["page_spans"]
                 if max(start, span["start"]) < min(end, span["end"])]
        if not pages or (m.get("page_start"), m.get("page_end"), m.get("page_number")) != (pages[0], pages[-1], pages[0]):
            raise ValueError("子片段页码与父级字符映射不一致。")
        if m.get("page_numbers") != ",".join(map(str, pages)) or m.get("page") != pages[0] - 1:
            raise ValueError("子片段的页面列表或索引错误。")
        chunk_id = make_chunk_id(child, i)

        expected_id = hashlib.sha256(json.dumps(
            [parent_id, start, end, child.page_content[:prefix]],
            ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        if chunk_id != expected_id or chunk_id in seen_ids:
            raise ValueError("子片段 ID 不匹配或重复。")
        seen_ids.add(chunk_id)
    if referenced_parents != set(parent_map):
        raise ValueError("存在没有子片段的父级，请重新检查结构化结果。")


def load_structured_documents(pdf_path: Path) -> tuple[list[Document], dict[str, Document], dict, bytes]:
    pdf_path = Path(pdf_path).resolve()
    manifest = _read_json(STRUCTURED_DIR / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("split_version") != SPLIT_VERSION:
        raise ValueError("不支持该结构化结果版本，请运行新版 document_processing。")
    if Path(manifest.get("pdf_path", "")).resolve() != pdf_path:
        raise ValueError("manifest 中的 PDF 来源与本次资料不一致。")
    if not pdf_path.is_file() or _file_sha256(pdf_path) != manifest.get("pdf_sha256"):
        raise ValueError("PDF 内容已变化，请重新生成结构化结果。")
    actual_ocr = {p.name: _file_sha256(p) for p in sorted(OCR_DIR.glob("page_*.txt"))}
    if not actual_ocr or actual_ocr != manifest.get("ocr_files"):
        raise ValueError("OCR 页面文件有增加、缺失或修改，请重新生成结构化结果。")
    if type(manifest.get("chunk_size")) is not int or type(manifest.get("chunk_overlap")) is not int:
        raise ValueError("manifest 缺少切分参数。")
    if not 0 <= manifest["chunk_overlap"] < manifest["chunk_size"]:
        raise ValueError("manifest 切分参数不合法。")
    chunks_data = (STRUCTURED_DIR / "chunks.json").read_bytes()
    parents_data = (STRUCTURED_DIR / "parents.json").read_bytes()
    for label, data in (("chunks", chunks_data), ("parents", parents_data)):
        if hashlib.sha256(data).hexdigest() != manifest.get(f"{label}_sha256"):
            raise ValueError(f"{label}.json 与 manifest 校验值不一致，请重新生成。")
    chunks = _decode_documents(chunks_data, "子片段")
    parent_map = _make_parent_map(_decode_documents(parents_data, "父级资料"))
    for doc in [*chunks, *parent_map.values()]:
        if Path(doc.metadata.get("source", "")).resolve() != pdf_path:
            raise ValueError("片段来源不是当前 PDF。")
    _validate_relationships(chunks, parent_map)
    return chunks, parent_map, manifest, parents_data


def ingest_documents(chunks: list[Document], store: Chroma, batch_size: int = BATCH_SIZE) -> dict:
    if not chunks or batch_size < 1:
        raise ValueError("片段不能为空，batch_size 必须大于 0。")
    if any(not doc.page_content.strip() for doc in chunks):
        raise ValueError("输入中有空片段。")
    ids = [make_chunk_id(doc, i) for i, doc in enumerate(chunks)]
    if len(ids) != len(set(ids)):
        raise ValueError("输入有重复 ID。")
    added = skipped = 0
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    for start in range(0, len(chunks), batch_size):
        batch_ids = ids[start:start + batch_size]
        batch_docs = chunks[start:start + batch_size]
        existing = set(store.get(ids=batch_ids, include=[])["ids"])
        new_pairs = [(doc, doc_id) for doc, doc_id in zip(batch_docs, batch_ids) if doc_id not in existing]
        skipped += len(batch_ids) - len(new_pairs)
        if new_pairs:
            print(f"第 {start // batch_size + 1}/{total_batches} 批：新增 {len(new_pairs)} 个子片段", flush=True)
            store.add_documents(documents=[p[0] for p in new_pairs], ids=[p[1] for p in new_pairs])
            added += len(new_pairs)
        else:
            print(f"第 {start // batch_size + 1}/{total_batches} 批：全部已存在，跳过")
    verified = set()
    for start in range(0, len(ids), batch_size):
        verified.update(store.get(ids=ids[start:start + batch_size], include=[])["ids"])
    if verified != set(ids):
        raise RuntimeError("子片段入库校验未通过，可重新运行以继续未完成的入库。")
    return {"total": len(chunks), "added": added, "skipped": skipped, "verified": len(verified)}


def get_index_status(store: Chroma) -> dict:
    count = len(store.get(include=[])["ids"])
    return {"chunk_count": count, "is_empty": count == 0}


def load_index_manifest(collection_name: str = COLLECTION_NAME) -> dict:
    _validate_collection_name(collection_name)
    version_dir = INDEX_ROOT / collection_name
    record = _read_json(version_dir / "index_manifest.json")
    if (not isinstance(record, dict)
            or record.get("collection_name") != collection_name
            or record.get("state") != "ready"):
        raise ValueError("该集合尚未完整构建，请先运行知识库入库。")
    for key in ("chunk_count", "parent_count"):
        if type(record.get(key)) is not int or record[key] < 1:
            raise ValueError("该集合的版本记录不完整，请重新入库。")
    return record


def load_parent_documents(collection_name: str = COLLECTION_NAME) -> dict[str, Document]:
    record = load_index_manifest(collection_name)
    version_dir = INDEX_ROOT / collection_name
    data = (version_dir / "parents.json").read_bytes()
    if hashlib.sha256(data).hexdigest() != record["identity"]["parents_sha256"]:
        raise ValueError("该集合的父级快照已改变，停止使用以避免引用错配。")
    parent_map = _make_parent_map(_decode_documents(data, "父级快照"))
    if len(parent_map) != record["parent_count"]:
        raise ValueError("父级快照数量与版本记录不一致。")
    return parent_map


def build_knowledge_base(pdf_path: Path, collection_name: str) -> dict:
    _validate_collection_name(collection_name)
    if collection_name == LEGACY_COLLECTION_NAME:
        raise ValueError("不能把新版资料写入旧 v1 集合，请使用 v2 或新的集合名。")
    chunks, parent_map, manifest, parents_data = load_structured_documents(pdf_path)
    identity = {key: manifest[key] for key in (
        "pdf_sha256", "chunks_sha256", "parents_sha256", "split_version", "chunk_size", "chunk_overlap"
    )}
    identity.update({"embedding_model": EMBEDDING_MODEL, "embedding_base_url": EMBEDDING_BASE_URL})
    version_dir = INDEX_ROOT / collection_name
    record_path = version_dir / "index_manifest.json"
    record = _read_json(record_path) if record_path.is_file() else None
    if record is not None:
        if record.get("collection_name") != collection_name or record.get("identity") != identity:
            raise ValueError("集合已绑定其他资料或模型版本，请使用新的集合名称，不要混合入库。")
        if _file_sha256(version_dir / "parents.json") != identity["parents_sha256"]:
            raise ValueError("集合父级快照校验失败。")


    store = open_vector_store(collection_name)
    expected_ids = {make_chunk_id(doc, i) for i, doc in enumerate(chunks)}
    existing_ids = set(store.get(include=[])["ids"])
    if existing_ids and record is None:
        raise ValueError("集合已有数据却没有版本记录，请换一个新的集合名。")
    if existing_ids - expected_ids:
        raise ValueError("集合中存在其他版本的片段，请使用新的集合名。")
    if record is None:
        _atomic_write(version_dir / "parents.json", parents_data)
    record = {
        "collection_name": collection_name, "identity": identity, "state": "building",
        "chunk_count": len(chunks), "parent_count": len(parent_map),
    }

    _save_json(record_path, record)
    report = ingest_documents(chunks, store)
    status = get_index_status(store)
    if status["chunk_count"] != len(chunks):
        raise RuntimeError("集合数量与输入片段数不一致。")
    record["state"] = "ready"
    _save_json(record_path, record)
    return {
        "file_name": Path(pdf_path).name, "collection_name": collection_name,
        "page_count": len(manifest["ocr_files"]), "chunk_count": len(chunks),
        "parent_count": len(parent_map), "added": report["added"], "skipped": report["skipped"],
        "stored_count": status["chunk_count"], "parent_snapshot": str(version_dir / "parents.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="从 structured_v2 构建父子片段知识库。")
    parser.add_argument("--check-only", action="store_true", help="仅校验文件与父子关联，不创建数据库、不调用模型")
    parser.add_argument("--collection-name", default=COLLECTION_NAME)
    args = parser.parse_args()
    if args.check_only:
        chunks, parents, manifest, _ = load_structured_documents(PDF_PATH)
        print(f"校验通过：{len(manifest['ocr_files'])} 页，{len(chunks)} 个子片段，{len(parents)} 份父级资料。")
        return
    report = build_knowledge_base(PDF_PATH, args.collection_name)
    print("\n===== 第二版知识库构建完成 =====")
    print(f"集合名称：{report['collection_name']}")
    print(f"页面数量：{report['page_count']}")
    print(f"子片段数量：{report['chunk_count']}")
    print(f"父级资料数量：{report['parent_count']}（保存在本地，不重复向量化）")
    print(f"本次新增：{report['added']}；已存在并跳过：{report['skipped']}")
    print(f"集合已保存片段：{report['stored_count']}")
    print(f"父级快照：{report['parent_snapshot']}")
    print("下一步在上下文选择时按 parent_id 找父级，并使用父级自身的页码范围。")


if __name__ == "__main__":
    main()
