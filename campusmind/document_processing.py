# 文档处理模块：保留旧切分接口；主入口生成奖学金章节的父子片段。

from config import (
    PROJECT_DIR,
    PDF_PATH,
    PROCESSED_DIR,
    OCR_DIR,
    STRUCTURED_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def preview_pdf_ocr(
    pdf_path: Path,
    output_dir: Path,
    page_numbers: list[int] | None = None,
) -> None:
    """识别 PDF 页面并保存图片和文字，不指定页码则处理全部页面。"""
    import pymupdf
    from rapidocr import RapidOCR

    if not pdf_path.is_file():
        raise FileNotFoundError(f"找不到 PDF 文件：{pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    engine = RapidOCR()

    with pymupdf.open(pdf_path) as pdf:
        print(f"PDF 总页数：{len(pdf)}")

        pages_to_process = (
            page_numbers
            if page_numbers is not None
            else range(1, len(pdf) + 1)
        )

        for page_number in pages_to_process:
            if not 1 <= page_number <= len(pdf):
                raise ValueError(f"页码超出范围：{page_number}")

            print(f"\n正在识别 PDF 第 {page_number} 页……")

            page = pdf[page_number - 1]
            pixmap = page.get_pixmap(
                dpi=150,
                colorspace=pymupdf.csRGB,
                alpha=False,
            )

            image_path = output_dir / f"page_{page_number:03d}.png"
            pixmap.save(str(image_path))
            del pixmap

            result = engine(str(image_path))
            lines = result.txts if result.txts is not None else ()
            text = "\n".join(lines)

            text_path = output_dir / f"page_{page_number:03d}.txt"
            text_path.write_text(text, encoding="utf-8")

            if text.strip():
                print(f"识别完成：{len(text)} 个字符")
                print(f"文字保存到：{text_path}")
            else:
                print("本页没有识别出文字，需要检查对应图片。")

            del result


def load_ocr_documents(
    ocr_dir: Path,
    pdf_path: Path,
) -> list[Document]:
    """读取逐页 OCR 文本，附上来源和 PDF 页码。"""
    if not ocr_dir.is_dir():
        raise FileNotFoundError(f"OCR 结果目录不存在：{ocr_dir}")

    text_files = [
        path
        for path in ocr_dir.glob("page_*.txt")
        if path.is_file()
        and path.stem.removeprefix("page_").isdigit()
    ]

    if not text_files:
        raise FileNotFoundError(f"目录中没有 OCR 页面文本：{ocr_dir}")

    text_files.sort(
        key=lambda path: int(path.stem.removeprefix("page_"))
    )

    documents = []

    for text_path in text_files:
        page_number = int(text_path.stem.removeprefix("page_"))

        if page_number < 1:
            raise ValueError(f"页面文件的页码不正确：{text_path.name}")

        documents.append(
            Document(
                page_content=text_path.read_text(encoding="utf-8"),
                metadata={
                    "source": str(pdf_path.resolve()),
                    "file_name": pdf_path.name,
                    "page": page_number - 1,
                    "page_number": page_number,
                    "extraction_method": "ocr",
                },
            )
        )

    return documents


def check_extraction(
    documents: list[Document],
    total_pages: int,
    min_chars: int = 30,
) -> dict:
    """检查页面完整性和文字量，不修改原始内容。"""
    page_numbers = [
        doc.metadata["page_number"] for doc in documents
    ]

    expected_pages = set(range(1, total_pages + 1))
    actual_pages = set(page_numbers)
    page_counts = Counter(page_numbers)

    empty_pages = []
    short_pages = []
    total_chars = 0

    for doc in documents:
        page_number = doc.metadata["page_number"]
        char_count = len("".join(doc.page_content.split()))
        total_chars += char_count

        if char_count == 0:
            empty_pages.append(page_number)
        elif char_count < min_chars:
            short_pages.append(page_number)

    report = {
        "total_pages": total_pages,
        "loaded_documents": len(documents),
        "missing_pages": sorted(expected_pages - actual_pages),
        "unexpected_pages": sorted(actual_pages - expected_pages),
        "duplicate_pages": sorted(
            page for page, count in page_counts.items() if count > 1
        ),
        "empty_pages": sorted(empty_pages),
        "short_pages": sorted(short_pages),
        "total_chars": total_chars,
    }

    print("\n===== OCR 结果检查 =====")
    print(f"PDF 总页数：{total_pages}")
    print(f"已加载页面 Document 数：{len(documents)}")
    print(f"总文字量（不含空白）：{total_chars}")
    print(f"缺失页：{report['missing_pages']}")
    print(f"超出范围的页码：{report['unexpected_pages']}")
    print(f"重复页：{report['duplicate_pages']}")
    print(f"无文字页：{report['empty_pages']}")
    print(
        f"少于 {min_chars} 个字符的非空页："
        f"{report['short_pages']}"
    )

    return report


def clean_documents(
    documents: list[Document],
) -> list[Document]:
    """保守清洗空白字符，保留原文信息和元数据。"""
    cleaned_documents = []

    for doc in documents:
        text = doc.page_content
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\ufeff", "").replace("\u200b", "")
        text = text.replace("\u00a0", " ").replace("\u3000", " ")

        cleaned_lines = []
        previous_line_empty = False

        for line in text.split("\n"):
            line = line.strip()

            if line:
                cleaned_lines.append(line)
                previous_line_empty = False
            elif not previous_line_empty:
                cleaned_lines.append("")
                previous_line_empty = True

        cleaned_documents.append(
            Document(
                page_content="\n".join(cleaned_lines).strip(),
                metadata=doc.metadata.copy(),
            )
        )

    return cleaned_documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """兼容旧版 knowledge_base.py 的逐页切分；新版使用 split_child_documents。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap 必须大于等于 0，且小于 chunk_size"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # 依次尝试空行、句末标点、换行等边界
        separators=[
            "\n\n", "。", "！", "？", "\n", "；", "，", " ", ""
        ],
        keep_separator="end",
        is_separator_regex=False,
    )

    # 空白页仍保留在页面数据中，但不生成检索片段
    nonempty_documents = [
        doc for doc in documents if doc.page_content.strip()
    ]

    chunks = splitter.split_documents(nonempty_documents)

    page_chunk_counts = Counter()

    for chunk_index, chunk in enumerate(chunks):
        page_key = (
            chunk.metadata["source"],
            chunk.metadata["page_number"],
        )
        page_chunk_counts[page_key] += 1

        # 全部片段的顺序编号，从 0 开始
        chunk.metadata["chunk_index"] = chunk_index

        # 当前页内的片段编号，从 1 开始
        chunk.metadata["page_chunk_number"] = page_chunk_counts[page_key]
        chunk.metadata["char_count"] = len(chunk.page_content)

    return chunks


def save_chunks(
    chunks: list[Document],
    output_path: Path,
) -> None:
    """将片段文字和元数据一起保存为 JSON。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# 以下章节范围已经对照当前手册的 OCR 和 PDF 正文确认。
# 数字都是 PDF 阅读器页码，不是目录中的印刷页码。
# 第一阶段只为四类奖学金确认章节边界，其余页面不跨页合并。
SCHOLARSHIP_SECTIONS = [
    {"section_id": "national_scholarship", "scholarship_type": "国家奖学金",
     "section_title": "湖南信息学院国家奖学金评审办法", "page_start": 119, "page_end": 124},
    {"section_id": "national_inspirational", "scholarship_type": "国家励志奖学金",
     "section_title": "湖南信息学院国家励志奖学金评审办法", "page_start": 125, "page_end": 125},
    {"section_id": "yingcai_scholarship", "scholarship_type": "英才奖学金",
     "section_title": "湖南信息学院英才奖学金评审办法", "page_start": 126, "page_end": 128},
    {"section_id": "xiangxin_inspirational", "scholarship_type": "湘信励志奖学金",
     "section_title": "湖南信息学院湘信励志奖学金评审办法", "page_start": 129, "page_end": 131},
]

# 人工抽查发现的问题只登记、不猜测修正文；此列表不是全书质量检查结果。
OCR_REVIEW_NOTES = {
    122: "申请表：纯文本 OCR 未保留表格行列关系，使用前对照原页。",
    123: "申请表续页：纯文本 OCR 未保留表格行列关系。",
    124: "填写说明第 3 项附近有 OCR 残缺（关申电），需对照原页。",
    126: "第二条及第六条第（一）项 OCR 明显残缺，不能凭空补全。",
    131: "申请表：纯文本 OCR 未保留表格行列关系，使用前对照原页。",
}


def _stable_id(*parts) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _remove_verified_margins(doc: Document, audit: list[dict]) -> str:
    """仅用于已核对的奖学金页；只去掉边缘的已知校徽文字和横线页码。"""
    lines = doc.page_content.splitlines()
    nonempty = [i for i, line in enumerate(lines) if line.strip()]
    removed = set()
    for index in nonempty[:2]:
        line = lines[index].strip()
        compact = "".join(line.split())
        is_school = compact in {"湖南信息学院", "湖南信息學院", "湖南信息尊院"}
        is_english = bool(re.search(r"University\s+of\s+Informa", line, re.I))
        if is_school or is_english:
            removed.add(index)
    if nonempty and re.fullmatch(r"[-—–－]\s*\d+\s*[-—–－]", lines[nonempty[-1]].strip()):
        removed.add(nonempty[-1])
    for index in sorted(removed):
        audit.append({"page_number": doc.metadata["page_number"],
                      "line_number": index + 1, "text": lines[index],
                      "reason": "已确认的页边校徽文字或横线页码"})
    return "\n".join(line for i, line in enumerate(lines) if i not in removed).strip()


def build_sections(pages: list[Document]) -> tuple[list[Document], list[dict]]:
    """奖学金按已确认的制度范围合并，其余内容逐页保留，记录页码映射。"""
    if not pages:
        raise ValueError("没有页面可供组织章节。")
    if len({p.metadata['source'] for p in pages}) != 1:
        raise ValueError("本函数一次只处理一本手册。")
    by_page = {p.metadata["page_number"]: p for p in pages}
    if len(by_page) != len(pages):
        raise ValueError("页面编号重复。")
    assigned = {}
    for spec in SCHOLARSHIP_SECTIONS:
        for number in range(spec["page_start"], spec["page_end"] + 1):
            if number not in by_page:
                raise ValueError(f"缺少章节所需的 PDF 第 {number} 页。")
            if number in assigned:
                raise ValueError("章节范围发生重叠。")
            assigned[number] = spec
        first_lines = by_page[spec["page_start"]].page_content.splitlines()[:8]
        if not any("".join(line.split()) == spec["section_title"] for line in first_lines):
            raise ValueError(f"{spec['section_title']} 的起始标题不匹配，请重新确认章节范围。")
    # 防止将下一份助学金制度误并入奖学金。
    if 132 not in by_page or "湖南信息学院国家助学金管理实施办法" not in by_page[132].page_content:
        raise ValueError("第 132 页的下一制度标题不匹配，请重新确认章节范围。")

    sections, audit = [], []
    for number in sorted(by_page):
        spec = assigned.get(number)
        if spec and number != spec["page_start"]:
            continue
        selected = range(spec["page_start"], spec["page_end"] + 1) if spec else [number]
        parts, spans, position = [], [], 0
        for page_number in selected:
            page = by_page[page_number]
            body = _remove_verified_margins(page, audit) if spec else page.page_content.strip()
            if not body:
                continue
            if parts:
                position += 1  # 页面之间保留换行，不猜测表格或段落连接方式
            spans.append({"page_number": page_number, "start": position, "end": position + len(body)})
            parts.append(body)
            position += len(body)
        if not parts:
            continue
        metadata = by_page[number].metadata.copy()
        metadata.update({
            "section_id": spec["section_id"] if spec else f"unclassified_page_{number:03d}",
            "section_title": spec["section_title"] if spec else f"未归类页面（PDF 第 {number} 页）",
            "scholarship_type": spec["scholarship_type"] if spec else "",
            "structure_scope": "verified_scholarship" if spec else "page_fallback",
            "page_start": spans[0]["page_number"], "page_end": spans[-1]["page_number"],
            "page_spans": spans,
        })
        sections.append(Document(page_content="\n".join(parts), metadata=metadata))
    return sections, audit


def _slice_with_pages(doc: Document, start: int, end: int) -> Document:
    """按字符范围取内容，同时把范围准确映射回 PDF 页码。"""
    while start < end and doc.page_content[start].isspace():
        start += 1
    while end > start and doc.page_content[end - 1].isspace():
        end -= 1
    spans = []
    for span in doc.metadata["page_spans"]:
        left, right = max(start, span["start"]), min(end, span["end"])
        if left < right:
            spans.append({"page_number": span["page_number"], "start": left - start, "end": right - start})
    if not spans:
        raise ValueError("切分范围没有有效页面映射。")
    metadata = doc.metadata.copy()
    metadata.update({"page_spans": spans, "page_start": spans[0]["page_number"],
                     "page_end": spans[-1]["page_number"], "page_number": spans[0]["page_number"],
                     "page": spans[0]["page_number"] - 1})
    return Document(page_content=doc.page_content[start:end], metadata=metadata)


def build_parent_documents(sections: list[Document]) -> list[Document]:
    """按章及附件组织父级资料，允许跨页，不跨奖学金制度；无章标题的制度整体保留。"""
    parents = []
    for section in sections:
        text = section.page_content
        boundaries = {0: "正文"}
        if section.metadata["structure_scope"] == "verified_scholarship":
            page_starts = {span["start"] for span in section.metadata["page_spans"]}
            lines = text.splitlines(keepends=True)
            offset = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if re.match(r"^第[一二三四五六七八九十百零〇两0-9]+章", stripped):
                    title = stripped
                    # 如“第二章”和“奖励标准与评选条件”被 OCR 分为两行
                    if re.fullmatch(r"第[一二三四五六七八九十百零〇两0-9]+章", title) and i + 1 < len(lines):
                        title += " " + lines[i + 1].strip()
                    boundaries[offset] = title
                elif offset in page_starts and (stripped.startswith(("附件：", "附件:")) or re.match(r"^《.+》填写说明", stripped)):
                    title = stripped
                    if stripped in {"附件：", "附件:"} and i + 1 < len(lines):
                        title += lines[i + 1].strip()
                    boundaries[offset] = title
                offset += len(line)
        starts = sorted(boundaries)
        for index, start in enumerate(starts):
            end = starts[index + 1] if index + 1 < len(starts) else len(text)
            if not text[start:end].strip():
                continue
            parent = _slice_with_pages(section, start, end)
            parent_id = _stable_id(section.metadata["source"], section.metadata["section_id"], start, parent.page_content)
            page_numbers = [span["page_number"] for span in parent.metadata["page_spans"]]
            review_pages = [number for number in page_numbers if number in OCR_REVIEW_NOTES]
            parent.metadata.update({
                "parent_id": parent_id, "parent_title": boundaries[start],
                "parent_index": len(parents), "char_count": len(parent.page_content),
                "needs_ocr_review": bool(review_pages),
                "ocr_review_pages": ",".join(map(str, review_pages)),
            })
            parents.append(parent)
    return parents


def split_child_documents(
    parents: list[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """小片段用于检索；parent_id 用于随后扩展完整父级上下文。"""
    chunks = []
    page_counts = Counter()
    for parent in parents:
        prefix = ""
        if parent.metadata["structure_scope"] == "verified_scholarship":
            prefix = f"[所属制度：{parent.metadata['section_title']}｜{parent.metadata['parent_title']}]\n"
        body_size = chunk_size - len(prefix)
        if not 0 <= chunk_overlap < body_size:
            raise ValueError("扣除章节提示后，片段长度必须大于重叠长度。")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_size, chunk_overlap=chunk_overlap, length_function=len,
            separators=["\n\n", "。", "！", "？", "\n", "；", "，", " ", ""],
            keep_separator="end", is_separator_regex=False, add_start_index=True,
        )
        pieces = splitter.create_documents([parent.page_content])
        for child_index, piece in enumerate(pieces):
            start = piece.metadata["start_index"]
            end = start + len(piece.page_content)
            if start < 0 or parent.page_content[start:end] != piece.page_content:
                raise ValueError("无法准确定位子片段，停止生成错误的来源页码。")
            child = _slice_with_pages(parent, start, end)
            metadata = child.metadata.copy()
            spans = metadata.pop("page_spans")  # Chroma 元数据保持为标量
            metadata.pop("parent_index", None)
            metadata["page_numbers"] = ",".join(str(s['page_number']) for s in spans)
            page_counts[metadata["page_number"]] += 1
            metadata.update({
                "chunk_index": len(chunks), "child_index": child_index,
                "page_chunk_number": page_counts[metadata["page_number"]],
                "parent_start_index": start, "parent_end_index": end,
                "section_prefix_length": len(prefix), "split_version": "scholarship_parent_child_v2",
                "chunk_id": _stable_id(metadata["parent_id"], start, end, prefix),
            })
            content = prefix + child.page_content
            metadata["char_count"] = len(content)
            chunks.append(Document(page_content=content, metadata=metadata))
    return chunks


def process_structured_documents(
    documents: list[Document], output_dir: Path,
    chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP,
) -> dict:
    """生成可检查的第二版资料；不修改 OCR，也不操作数据库。"""
    cleaned = clean_documents(documents)
    sections, margin_audit = build_sections(cleaned)
    parents = build_parent_documents(sections)
    chunks = split_child_documents(parents, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("没有生成有效片段。")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_chunks(parents, output_dir / "parents.json")
    save_chunks(chunks, output_dir / "chunks.json")
    report = {
        "split_version": "scholarship_parent_child_v2", "input_pages": len(documents),
        "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
        "parent_count": len(parents), "chunk_count": len(chunks),
        "verified_sections": SCHOLARSHIP_SECTIONS,
        "coverage_note": "只有奖学金范围按制度/章组织；其他页面仍为逐页父级，不代表全书已经结构化。",
        "known_ocr_review_notes": OCR_REVIEW_NOTES,
        "removed_margin_lines": margin_audit,
        "short_chunk_ids": [d.metadata['chunk_id'] for d in chunks if len(d.page_content) < 30],
        "long_parent_ids": [d.metadata['parent_id'] for d in parents if len(d.page_content) > 2500],
        "empty_pages": [d.metadata['page_number'] for d in cleaned if not d.page_content.strip()],
    }
    (output_dir / "processing_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    preview = ["奖学金父级资料预览（页码为 PDF 页面序号）", "已知 OCR 残缺仅标记，没有猜测补写。"]
    for parent in parents:
        if parent.metadata["structure_scope"] != "verified_scholarship":
            continue
        m = parent.metadata
        preview.extend(["", "=" * 60, f"{m['section_title']} / {m['parent_title']}",
                        f"PDF 第 {m['page_start']}–{m['page_end']} 页；{len(parent.page_content)} 字符",
                        f"parent_id：{m['parent_id']}", f"需核对 OCR：{m['needs_ocr_review']}", parent.page_content])
    (output_dir / "scholarship_preview.txt").write_text("\n".join(preview), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="生成奖学金父子片段，不调用模型或修改数据库。")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    pdf_path = PDF_PATH if project_dir == PROJECT_DIR else project_dir / "new_data" / "raw" / PDF_PATH.name
    processed_dir = PROCESSED_DIR if project_dir == PROJECT_DIR else project_dir / "new_data" / "processed"
    ocr_dir = OCR_DIR if project_dir == PROJECT_DIR else processed_dir / "ocr_pages"
    output_dir = args.output_dir or (STRUCTURED_DIR if project_dir == PROJECT_DIR else processed_dir / "structured_v2")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"找不到 PDF：{pdf_path}")
    documents = load_ocr_documents(ocr_dir, pdf_path)
    # 正常项目用已安装的 PyMuPDF；备用运行环境也可以用 pypdf 只读取页数。
    try:
        import pymupdf
    except ImportError:
        from pypdf import PdfReader
        total_pages = len(PdfReader(pdf_path).pages)
    else:
        with pymupdf.open(pdf_path) as pdf:
            total_pages = len(pdf)
    report = check_extraction(documents, total_pages)
    if any(report[key] for key in ("missing_pages", "unexpected_pages", "duplicate_pages")):
        raise ValueError("页面完整性检查未通过。")
    report = process_structured_documents(documents, output_dir)
    # 记录实际输入内容，未来入库时可核对版本。
    manifest = {
        "pdf_path": str(pdf_path),
        "pdf_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "ocr_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ocr_dir.glob('page_*.txt'))},
        "chunks_sha256": hashlib.sha256((output_dir / 'chunks.json').read_bytes()).hexdigest(),
        "parents_sha256": hashlib.sha256((output_dir / 'parents.json').read_bytes()).hexdigest(),
        "split_version": report['split_version'], "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print("\n===== 第二版文档处理完成 =====")
    for section in SCHOLARSHIP_SECTIONS:
        print(f"{section['scholarship_type']}：PDF 第 {section['page_start']}–{section['page_end']} 页")
    print(f"父级资料：{report['parent_count']}；检索片段：{report['chunk_count']}")
    print(f"已知需核对的 OCR 页：{list(OCR_REVIEW_NOTES)}")
    print(f"输出目录：{output_dir}")
    print("请先检查 scholarship_preview.txt。旧 chunks.json 和 Chroma 集合未修改。")
    print("knowledge_base.py 仍使用兼容的旧 split_documents；下一步再接入新片段和父级资料。")


if __name__ == "__main__":
    main()
