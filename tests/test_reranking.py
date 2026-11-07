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

    def test_oversized_parent_skipped_and_flagged(self):
        # B3：排名第一的父级超长也统一跳过，不再让整题失败。
        parents = {"a": parent("a", "123456"), "b": parent("b", "12")}
        result = r.select_context([child("a"), child("b")], max_chars=3, parent_map=parents)
        self.assertEqual([d.metadata["parent_id"] for d in result], ["b"])
        self.assertTrue(all(d.metadata["context_incomplete"] for d in result))

    def test_all_parents_oversized_returns_empty(self):
        result = r.select_context([child("a")], max_chars=2, parent_map={"a": parent("a", "123")})
        self.assertEqual(result, [])

    def test_no_truncation_flag_when_everything_fits(self):
        parents = {"a": parent("a", "1234")}
        result = r.select_context([child("a")], max_chars=10, parent_map=parents)
        self.assertFalse(result[0].metadata["context_incomplete"])

    def test_parse_rerank_results_maps_and_sorts(self):
        candidates = [
            Document(page_content="甲", metadata={"chunk_id": "a"}),
            Document(page_content="乙", metadata={"chunk_id": "b"}),
        ]
        response = {"results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.5},
        ]}
        ranked = r.parse_rerank_results(response, candidates)
        self.assertEqual([d.metadata["chunk_id"] for d in ranked], ["b", "a"])
        self.assertEqual([d.metadata["rerank_rank"] for d in ranked], [1, 2])
        self.assertEqual(ranked[0].metadata["rerank_input_index"], 1)
        self.assertEqual(candidates[0].metadata, {"chunk_id": "a"})

    def test_parse_rerank_results_rejects_invalid(self):
        candidates = [Document(page_content="甲"), Document(page_content="乙")]
        invalid_responses = (
            [],
            {},
            {"results": "bad"},
            {"results": [{"index": 0, "relevance_score": 0.5}]},  # 数量不完整
            {"results": [{"index": 0, "relevance_score": 0.5},
                         {"index": 0, "relevance_score": 0.4}]},  # 重复 index
            {"results": [{"index": 2, "relevance_score": 0.5},
                         {"index": 0, "relevance_score": 0.4}]},  # 越界
            {"results": [{"index": 0, "relevance_score": True},
                         {"index": 1, "relevance_score": 0.4}]},  # 布尔分数
            {"results": [{"index": 0, "relevance_score": float("nan")},
                         {"index": 1, "relevance_score": 0.4}]},  # NaN
        )
        for response in invalid_responses:
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    r.parse_rerank_results(response, candidates)

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
