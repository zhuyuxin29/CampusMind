import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document
from campusmind import document_processing as d


SECTION_TITLES = {
    119: "湖南信息学院国家奖学金评审办法",
    125: "湖南信息学院国家励志奖学金评审办法",
    126: "湖南信息学院英才奖学金评审办法",
    129: "湖南信息学院湘信励志奖学金评审办法",
}


def make_page(number, text):
    return Document(page_content=text, metadata={
        "source": "test.pdf", "file_name": "test.pdf",
        "page_number": number, "page": number - 1,
    })


def make_pages(empty=(), start=119, end=132):
    pages = []
    for number in range(start, end + 1):
        if number in empty:
            text = ""
        elif number == 132:
            text = "湖南信息学院国家助学金管理实施办法\n第一条 为规范国家助学金管理。"
        elif number in SECTION_TITLES:
            text = SECTION_TITLES[number] + "\n本章正文：申请条件与评审程序。"
        else:
            text = f"第 {number} 页的一般正文内容。"
        pages.append(make_page(number, text))
    return pages


class DocumentProcessingTests(unittest.TestCase):
    def test_split_version_single_source(self):
        # B7：SPLIT_VERSION 只从 config 导入，不再写死在文档处理模块。
        from config import SPLIT_VERSION
        self.assertEqual(d.SPLIT_VERSION, SPLIT_VERSION)
        source = Path(d.__file__).read_text(encoding="utf-8")
        self.assertNotIn('"scholarship_parent_child_v2"', source)

    def test_empty_scholarship_page_fails_fast(self):
        # B4：奖学金章节的空页不能静默跳过，否则条款会莫名缺失。
        for number in (119, 121, 125, 129, 131):
            with self.subTest(page=number):
                with self.assertRaisesRegex(ValueError, "没有 OCR 文字"):
                    d.build_sections(make_pages(empty={number}))

    def test_empty_other_page_is_allowed(self):
        pages = [make_page(100, "")] + make_pages()
        sections, _ = d.build_sections(pages)
        page_numbers = {section.metadata["page_start"] for section in sections}
        self.assertNotIn(100, page_numbers)
        self.assertTrue(any(s.metadata["section_id"] == "national_scholarship"
                            for s in sections))

    def test_processing_report_uses_split_version(self):
        pages = make_pages()
        with tempfile.TemporaryDirectory() as directory:
            report = d.process_structured_documents(pages, Path(directory))
        self.assertEqual(report["split_version"], d.SPLIT_VERSION)
        self.assertGreater(report["parent_count"], 0)
        self.assertGreater(report["chunk_count"], 0)


if __name__ == "__main__":
    unittest.main()
