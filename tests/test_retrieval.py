import json
import sys
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from campusmind import knowledge_base as kb, query_processing as q, reranking as r, retrieval as ret
from campusmind import generation as g

ROOT = Path(__file__).resolve().parents[1]


def parent(key, text, start=1, end=2):
    return Document(page_content=text, metadata={"parent_id": key, "source": "test.pdf",
        "section_id": "section", "scholarship_type": "国家奖学金", "section_title": "测试制度",
        "page_start": start, "page_end": end, "page_number": start,
        "page_spans": [{"page_number": start, "start": 0, "end": len(text)}]})


def child(key, score=0.9):
    return Document(page_content="子片段", metadata={"parent_id": key, "source": "test.pdf",
        "section_id": "section", "scholarship_type": "国家奖学金", "page_number": 2,
        "rerank_score": score, "rerank_rank": 1, "chunk_id": key + "-child"})



class ModuleTests(unittest.TestCase):
    def test_display_range(self):
        self.assertIn("测试制度｜PDF 第 1–2 页", ret.format_document_source(parent("a", "正文")))

    def test_v2_chunk_id_without_legacy_index(self):
        doc = Document(page_content="正文", metadata={"split_version": kb.SPLIT_VERSION, "chunk_id": "a" * 64})
        self.assertEqual(ret._get_chunk_id(doc), "a" * 64)
        self.assertEqual(len(ret.fuse_rankings([[doc, doc], [doc]])), 1)
