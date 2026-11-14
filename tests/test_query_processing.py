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
    def test_generic_scholarship_stops_before_api(self):
        for question in ("奖学金如何申请？", "励志奖学金需要哪些材料？", "奖学金申请条件有哪些？"):
            with self.subTest(question=question), patch.object(q, "_request_json") as request:
                with self.assertRaises(q.ClarificationNeeded):
                    q.contextualize_question(question, [])
                request.assert_not_called()

    def test_specific_overview_comparison_not_blocked(self):
        for question in ("国家奖学金如何申请？", "有哪些奖学金？", "奖学金有哪些？", "各类奖学金怎么申请？", "国家与英才奖学金有什么区别？", "缓考需要哪些材料？"):
            self.assertFalse(q.needs_scholarship_clarification(question), question)

    def test_named_scholarship_not_forced_to_clarify(self):
        for question in ("校级奖学金如何申请？", "企业奖学金申请条件有哪些？",
                         "校友奖学金怎么申请？", "社会奖学金需要哪些材料？"):
            self.assertFalse(q.needs_scholarship_clarification(question), question)

    def test_bare_scholarship_still_clarifies(self):
        for question in ("奖学金如何申请？", "请问奖学金怎么申请？",
                         "申请奖学金需要哪些材料？", "关于奖学金的条件"):
            self.assertTrue(q.needs_scholarship_clarification(question), question)

    def test_ambiguous_alias_still_clarifies(self):
        self.assertTrue(q.needs_scholarship_clarification("励志奖学金需要哪些材料？"))
        self.assertFalse(q.needs_scholarship_clarification("国家励志奖学金需要哪些材料？"))

    def test_validate_question(self):
        for bad in (None, 123, [], {}):
            with self.assertRaises(ValueError):
                q.validate_question(bad)
        for bad in ("", "   ", "\n\t", "？？？", "！！！", "😀😀😀", "……"):
            with self.subTest(question=bad):
                with self.assertRaises(ValueError):
                    q.validate_question(bad)
        with self.assertRaisesRegex(ValueError, "不能超过"):
            q.validate_question("字" * (q.MAX_QUESTION_CHARS + 1))
        self.assertEqual(q.validate_question("  缓考需要什么材料？  "), "缓考需要什么材料？")
        self.assertEqual(q.validate_question("A1"), "A1")

    def test_history_user_selection_and_postcheck(self):
        response = {"standalone_question": "国家奖学金申请需要什么材料？", "needs_clarification": False}
        with patch.object(q, "_request_json", return_value=response):
            self.assertEqual(q.contextualize_question("奖学金要什么材料？", [{"question": "国家奖学金怎么申请？", "answer": "旧回答"}]), response["standalone_question"])
        with patch.object(q, "_request_json") as request:
            with self.assertRaises(q.ClarificationNeeded):
                q.contextualize_question("奖学金怎么申请？", [{"question": "你好", "answer": "国家奖学金"}])
            request.assert_not_called()
        response["standalone_question"] = "奖学金如何申请？"
        with patch.object(q, "_request_json", return_value=response):
            with self.assertRaises(q.ClarificationNeeded):
                q.contextualize_question("怎么申请？", [])
