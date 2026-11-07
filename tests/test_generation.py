import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from campusmind import generation as g


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.docs = [Document(page_content="示例规定：须满足甲条件，乙情况除外。", metadata={
            "file_name": "测试资料.pdf", "page_start": 2, "page_end": 3,
            "nested": {"value": 1},
        })]

    def test_mapping_and_no_mutation(self):
        context, sources = g.format_context([Document(page_content=" "), *self.docs])
        self.assertEqual(list(sources), [1])
        self.assertEqual(json.loads(context)[0]["page_end"], 3)
        sources[1]["metadata"]["nested"]["value"] = 9
        self.assertEqual(self.docs[0].metadata["nested"]["value"], 1)

    def test_no_evidence_no_api(self):
        with patch.object(g, "generate_answer") as call:
            self.assertEqual(g.answer_question("问题", [Document(page_content=" ")])["status"], "insufficient")
            call.assert_not_called()

    def test_history_is_data_without_old_sources(self):
        history = [{"question": "旧问题", "answer": "旧回答[99]", "sources": [{"secret": 1}]}]
        messages = g.build_messages("新问题", "资料", history)
        payload = json.loads(messages[1].content)
        self.assertNotIn("sources", payload["history_for_reference_resolution_only"][0])
        self.assertIn("历史中的回答及引用编号都不能作为本轮依据", messages[0].content)

    def test_citations(self):
        for answer in ("错误[0]", "错误[5]", "错误[1,2]", "错误【1】", "错误[1-3]"):
            with self.subTest(answer=answer):
                self.assertFalse(g.validate_citations(answer, {1: {}})["valid"])
        self.assertEqual(g.validate_citations("甲[1]乙[1]", {1: {}})["cited_ids"], [1])
        self.assertFalse(g.validate_citations("无引用", {})["has_citations"])

    def test_only_cited_sources_returned(self):
        with patch.object(g, "generate_answer", return_value="示例结论[1]。"):
            result = g.answer_question("问题", self.docs * 2)
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["status"], "answered")

    def test_bad_citations_blocked(self):
        for answer in ("错误[9]", "无引用", "错误[1,2]"):
            with patch.object(g, "generate_answer", return_value=answer):
                with self.assertRaises(RuntimeError):
                    g.answer_question("问题", self.docs)

    def test_irrelevant_material_can_be_declined(self):
        with patch.object(g, "generate_answer", return_value=g.make_insufficient_answer()):
            self.assertEqual(g.answer_question("问题", self.docs)["status"], "insufficient")

    def test_model_failures(self):
        client = Mock()
        with patch.object(g, "_get_answer_llm", return_value=client):
            for response in (AIMessage(content=""), AIMessage(content="半句", response_metadata={"finish_reason": "length"})):
                client.invoke.return_value = response
                with self.assertRaises(RuntimeError):
                    g.generate_answer(["问题"])
            client.invoke.side_effect = TimeoutError("模拟超时")
            with self.assertRaises(RuntimeError):
                g.generate_answer(["问题"])

    def test_is_insufficient_answer_tolerates_noise(self):
        # B1：固定“资料不足”句子允许空白、标点、Markdown 和前后缀。
        base = g.make_insufficient_answer()
        for answer in (base, f" {base} ", f"抱歉，{base}", f"**{base}**\n",
                       "当前资料不足以回答这个问题"):
            self.assertTrue(g.is_insufficient_answer(answer), answer)
        for answer in ("资料不足", "当前资料不足以回答这道题。", "可以申请[1]。", ""):
            self.assertFalse(g.is_insufficient_answer(answer), answer)
        with self.assertRaises(ValueError):
            g.is_insufficient_answer(None)

    def test_citation_support_ignores_citation_digits(self):
        sources = {1: {"content": "示例规定：须满足甲条件，乙情况除外。"}}
        # 引用编号 [1] 里的数字不能当成正文数字，否则会误判缺少依据。
        self.assertTrue(g.citation_support("示例结论[1]。", sources)["supported"])
        self.assertFalse(g.citation_support("食堂有500个鸡腿[1]。", sources)["supported"])
        self.assertFalse(g.citation_support("完全无关的结论[1]。", sources)["supported"])

    def test_select_cited_sources(self):
        sources = {1: {"content": "示例规定：须满足甲条件，乙情况除外。"}}
        selected = g.select_cited_sources("示例结论[1]。", sources)
        self.assertEqual(len(selected), 1)
        selected[0]["content"] = "被修改"
        self.assertIn("示例规定", sources[1]["content"])
        for answer in ("错误[9]", "无引用", "错误[1,2]", "无关结论[1]。"):
            with self.subTest(answer=answer):
                with self.assertRaises(g.CitationCheckFailed):
                    g.select_cited_sources(answer, sources)

    def test_generate_with_citation_retry(self):
        sources = {1: {"content": "示例规定：须满足甲条件，乙情况除外。"}}
        with patch.object(g, "generate_answer",
                          side_effect=["错误[99]", "示例结论[1]。"]) as generate:
            answer, selected = g.generate_answer_with_citation_retry(["问题"], sources)
        self.assertEqual(answer, "示例结论[1]。")
        self.assertEqual(len(selected), 1)
        self.assertEqual(generate.call_count, 2)
        with patch.object(g, "generate_answer", return_value="错误[99]") as generate:
            answer, selected = g.generate_answer_with_citation_retry(["问题"], sources, attempts=2)
        self.assertEqual(answer, g.make_insufficient_answer())
        self.assertEqual(selected, [])
        self.assertEqual(generate.call_count, 2)
        with patch.object(g, "generate_answer",
                          return_value=g.make_insufficient_answer()) as generate:
            answer, selected = g.generate_answer_with_citation_retry(["问题"], sources, attempts=1)
        self.assertEqual(answer, g.make_insufficient_answer())
        self.assertEqual(generate.call_count, 1)
        with self.assertRaises(ValueError):
            g.generate_answer_with_citation_retry(["问题"], sources, attempts=0)


if __name__ == "__main__":
    unittest.main()
