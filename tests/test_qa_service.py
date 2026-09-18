import importlib.util
import json
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch, Mock

from langchain_core.documents import Document
from campusmind.session_manager import ChatSession
from campusmind import qa_service as q


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.session = ChatSession()
        self.service = q.QAService()
        self.doc = Document(page_content="测试原文", metadata={"file_name": "手册.pdf", "page_start": 2, "page_end": 4})
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.mocks = {}
        for name, value in {
            "contextualize_question": "完整测试问题", "rewrite_queries": [],
            "retrieve_candidates": [self.doc], "rerank_documents": [self.doc],
            "select_context": [self.doc], "generate_answer": "测试结论[1]。",
        }.items():
            self.mocks[name] = self.stack.enter_context(patch.object(q, name, return_value=value))
        self.resource = self.stack.enter_context(patch.object(self.service, "_get_resources", return_value=(Mock(), {})))

    def test_success_saved_once_with_pages_and_timings(self):
        result = self.service.ask(" 原始问题 ", self.session)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["sources"][0]["page_end"], 4)
        self.assertEqual(len(self.session.get_recent_history()), 1)
        self.assertEqual(self.session.get_recent_history()[0]["question"], "原始问题")
        self.assertTrue(all(value is not None and value >= 0 for value in result["timings_ms"].values()))
        json.dumps(result, ensure_ascii=False)

    def test_invalid_input_stops(self):
        result = self.service.ask(" ", self.session)
        self.assertEqual(result["status"], "invalid_input")
        self.resource.assert_not_called()
        self.assertIsNone(result["timings_ms"]["generate"])

    def test_clarification_not_saved(self):
        self.mocks["contextualize_question"].side_effect = q.ClarificationNeeded("哪种奖学金？")
        result = self.service.ask("奖学金如何申请？", self.session)
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["answer"], "哪种奖学金？")
        self.assertEqual(self.session.get_recent_history(), [])
        self.resource.assert_not_called()

    def test_no_candidates_skips_paid_stages(self):
        self.mocks["retrieve_candidates"].return_value = []
        result = self.service.ask("问题", self.session)
        self.assertEqual(result["status"], "insufficient")
        self.assertTrue(result["saved"])
        self.mocks["rerank_documents"].assert_not_called()
        self.mocks["generate_answer"].assert_not_called()

    def test_invalid_citations_and_missing_citations_blocked(self):
        for answer in ("错误引用[99]", "无引用的结论"):
            self.mocks["generate_answer"].return_value = answer
            result = self.service.ask("问题", self.session)
            self.assertEqual(result["error_stage"], "validate_citations")
            self.assertEqual(result["answer"], "")
            self.assertEqual(result["sources"], [])
        self.assertEqual(self.session.get_recent_history(), [])

    def test_api_failure_not_insufficient(self):
        self.mocks["generate_answer"].side_effect = RuntimeError("sensitive payload")
        result = self.service.ask("问题", self.session)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_stage"], "generate")
        self.assertNotIn("sensitive", json.dumps(result))
        self.assertEqual(self.session.get_recent_history(), [])

    def test_model_insufficient(self):
        self.mocks["generate_answer"].return_value = q.make_insufficient_answer()
        result = self.service.ask("问题", self.session)
        self.assertEqual(result["status"], "insufficient")
        self.assertTrue(result["saved"])
        self.assertIsNone(result["timings_ms"]["validate_citations"])

    def test_history_and_session_isolation(self):
        self.session.add_turn("旧问题", "旧回答")
        self.service.ask("追问", self.session)
        self.assertEqual(self.mocks["contextualize_question"].call_args.args[1][0]["question"], "旧问题")
        other = ChatSession()
        self.service.ask("新问题", other)
        self.assertEqual(self.mocks["contextualize_question"].call_args.args[1], [])
        self.assertEqual(len(self.session.get_recent_history()), 2)
        self.assertEqual(len(other.get_recent_history()), 1)

    def test_save_failure_does_not_report_success(self):
        with patch.object(self.session, "add_turn", side_effect=RuntimeError):
            result = self.service.ask("问题", self.session)
        self.assertFalse(result["saved"])
        self.assertEqual(result["error_stage"], "save_session")

    def test_initialization_cached_and_failure_retryable(self):
        service = q.QAService()
        store = Mock()
        store.get.return_value = {"ids": ["child"], "documents": ["原文"], "metadatas": [
            {"split_version": q.SPLIT_VERSION, "parent_id": "parent"}]}
        with patch.object(q, "load_parent_documents", return_value={"parent": self.doc}) as load, \
             patch.object(q, "open_vector_store", return_value=store), \
             patch.object(q, "build_bm25_retriever"), patch.object(q, "build_vector_retriever"), \
             patch.object(q, "HybridRetriever"):
            store.get.side_effect = RuntimeError("temporary")
            with self.assertRaises(RuntimeError):
                service._get_resources()
            self.assertIsNone(service._resources)
            store.get.side_effect = None
            first = service._get_resources()
            self.assertIs(first, service._get_resources())
            self.assertEqual(load.call_count, 2)


    def test_session_from_previous_module_load_keeps_history(self):
        # 同一份源码加载成新的类，模拟热更新前留在页面里的旧实例。
        import importlib.util
        import campusmind.session_manager as manager
        spec = importlib.util.spec_from_file_location("previous_session_manager", manager.__file__)
        previous = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(previous)
        old_session = previous.ChatSession()
        old_session.add_turn("原来的问题", "原来的回答")
        self.assertNotIsInstance(old_session, q.ChatSession)
        result = self.service.ask("继续提问", old_session)
        self.assertEqual(result["status"], "answered")
        self.assertTrue(result["saved"])
        self.assertEqual(len(old_session.get_recent_history()), 2)
        self.assertEqual(old_session.get_recent_history()[0]["question"], "原来的问题")
        self.assertEqual(self.mocks["contextualize_question"].call_args.args[1][0]["question"], "原来的问题")

    def test_invalid_session_still_rejected(self):
        for session in (None, {}, {"get_recent_history": []}):
            result = self.service.ask("问题", session)
            self.assertEqual(result["status"], "invalid_input")
            self.assertFalse(result["saved"])
        self.resource.assert_not_called()



if __name__ == "__main__":
    unittest.main()
