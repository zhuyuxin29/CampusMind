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
    def test_expand_dedup_rank_and_own_pages(self):
        parents = {"a": parent("a", "父级完整原文"), "b": parent("b", "第二份原文", 3, 4)}
        result = r.select_context([child("a"), child("a", .8), child("b", .7)], parent_map=parents)
        self.assertEqual([d.metadata["parent_id"] for d in result], ["a", "b"])
        self.assertEqual(result[0].page_content, "父级完整原文")
        self.assertEqual(result[0].metadata["page_number"], 1)
        result[0].metadata["page_spans"][0]["start"] = 99
        self.assertEqual(parents["a"].metadata["page_spans"][0]["start"], 0)

    def test_budget_skip_whole_parent_and_continue(self):
        parents = {key: parent(key, text) for key, text in (("a", "12345"), ("b", "123456"), ("c", "1234"))}
        result = r.select_context([child("a"), child("b"), child("c")], max_chars=9, parent_map=parents)
        self.assertEqual([d.metadata["parent_id"] for d in result], ["a", "c"])
        self.assertEqual(sum(len(d.page_content) for d in result), 9)

    def test_oversized_first_parent(self):
        with self.assertRaisesRegex(ValueError, "超过"):
            r.select_context([child("a")], max_chars=2, parent_map={"a": parent("a", "123")})

    def test_missing_mismatched_parent(self):
        with self.assertRaises(ValueError):
            r.select_context([child("missing")], parent_map={})
        mismatched = parent("a", "原文")
        mismatched.metadata["source"] = "different.pdf"
        with self.assertRaises(ValueError):
            r.select_context([child("a")], parent_map={"a": mismatched})

    def test_empty_and_invalid_limits(self):
        with patch.object(r, "load_parent_documents") as load:
            self.assertEqual(r.select_context([]), [])
            load.assert_not_called()
        for kwargs in ({"top_n": 0}, {"max_chars": 0}, {"top_n": True}):
            with self.assertRaises(ValueError):
                r.select_context([], **kwargs)

    def test_top_n_and_snapshot_errors(self):
        parents = {k: parent(k, "原文") for k in ("a", "b")}
        self.assertEqual(len(r.select_context([child("a"), child("b")], 1, parent_map=parents)), 1)
        with patch.object(r, "load_parent_documents", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(ValueError, "快照"):
                r.select_context([child("a")])
