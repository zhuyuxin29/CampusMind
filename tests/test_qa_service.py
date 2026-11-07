import importlib.util
import json
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch, Mock

from langchain_core.documents import Document
from campusmind.errors import ServiceError
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
        self.stack.enter_context(patch.object(q, "missing_api_keys", return_value=[]))

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

    def test_missing_api_keys_is_config_error(self):
        # 没填 .env 时直接提示配置问题，而不是误报网络/额度故障。
        with patch.object(q, "missing_api_keys", return_value=["DEEPSEEK_API_KEY"]):
            result = self.service.ask("问题", self.session)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_stage"], "config")
        self.assertIn("DEEPSEEK_API_KEY", result["message"])
        self.assertIn(".env", result["message"])
        self.assertFalse(result["saved"])
        self.resource.assert_not_called()

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

    def test_invalid_citations_degrade_to_insufficient_after_retry(self):
        # B1：引用校验失败不再进入 error；重试一次后降级为资料不足。
        for answer in ("错误引用[99]", "无引用的结论", "食堂有500个鸡腿[1]。"):
            with self.subTest(answer=answer):
                self.mocks["generate_answer"].reset_mock()
                self.mocks["generate_answer"].return_value = answer
                result = self.service.ask("问题", self.session)
                self.assertEqual(result["status"], "insufficient")
                self.assertEqual(result["insufficient_reason"], "citation_unverified")
                self.assertEqual(result["answer"], q.make_insufficient_answer())
                self.assertEqual(result["sources"], [])
                self.assertIsNone(result["error_stage"])
                self.assertEqual(self.mocks["generate_answer"].call_count, 2)
        self.assertEqual(len(self.session.get_recent_history()), 3)

    def test_citation_retry_can_recover(self):
        self.mocks["generate_answer"].side_effect = ["错误引用[99]", "测试结论[1]。"]
        result = self.service.ask("问题", self.session)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(self.mocks["generate_answer"].call_count, 2)
        self.assertEqual(len(result["sources"]), 1)
        self.assertIsNone(result["insufficient_reason"])

    def test_api_failure_not_insufficient(self):
        self.mocks["generate_answer"].side_effect = RuntimeError("sensitive payload")
        result = self.service.ask("问题", self.session)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_stage"], "generate")
        self.assertNotIn("sensitive", json.dumps(result))
        self.assertEqual(self.session.get_recent_history(), [])

    def test_error_message_mapped_by_exception_type(self):
        self.mocks["generate_answer"].side_effect = TimeoutError("timed out")
        result = self.service.ask("问题", self.session)
        self.assertEqual(result["status"], "error")
        self.assertIn("超时", result["message"])
        self.mocks["generate_answer"].side_effect = ConnectionError("offline")
        result = self.service.ask("问题", self.session)
        self.assertEqual(result["status"], "error")
        self.assertIn("网络", result["message"])

    def test_service_error_message_is_kept(self):
        # P1-2：项目内安全异常的中文说明不能被泛化成“网络/额度”提示。
        self.mocks["generate_answer"].side_effect = ServiceError(
            "模型未正常完成回答，请重试或缩小问题范围。"
        )
        result = self.service.ask("问题", self.session)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_stage"], "generate")
        self.assertEqual(result["message"], "模型未正常完成回答，请重试或缩小问题范围。")

    def test_model_insufficient(self):
        self.mocks["generate_answer"].return_value = q.make_insufficient_answer()
        result = self.service.ask("问题", self.session)
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["insufficient_reason"], "model_declined")
        self.assertTrue(result["saved"])
        self.assertIsNone(result["timings_ms"]["validate_citations"])

    def test_insufficient_detection_tolerates_noise(self):
        # B1：固定句子带前后缀、空白或 Markdown 装饰时仍按资料不足处理。
        self.mocks["generate_answer"].return_value = (
            " 抱歉，\n当前资料不足以回答这个问题，请补充具体事项或相关资料。 \n"
        )
        result = self.service.ask("问题", self.session)
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["answer"], q.make_insufficient_answer())
        self.assertIsNone(result["error_stage"])

    def test_punctuation_only_input_is_invalid(self):
        # B2：纯标点或 emoji 是输入问题，不能误报为网络/密钥故障。
        for question in ("？？？", "！！！", "😀😀😀", "。。。", "  !?  "):
            with self.subTest(question=question):
                result = self.service.ask(question, self.session)
                self.assertEqual(result["status"], "invalid_input")
                self.assertEqual(result["error_stage"], "validate")
        self.resource.assert_not_called()

    def test_context_incomplete_flagged(self):
        # B3：父级预算跳过时，结果带 context_incomplete 标记。
        self.mocks["select_context"].return_value = [Document(
            page_content="测试原文",
            metadata={"file_name": "手册.pdf", "page_start": 2, "page_end": 4,
                      "context_incomplete": True},
        )]
        result = self.service.ask("问题", self.session)
        self.assertEqual(result["status"], "answered")
        self.assertTrue(result["context_incomplete"])

    def test_context_incomplete_when_all_parents_skipped(self):
        # P1-3：所有父级都因预算被跳过、docs 为空时也要保留裁剪标记。
        def fake_select(ranked, **kwargs):
            kwargs["stats"].update(
                {"truncated": True, "selected_count": 0, "skipped_count": 1}
            )
            return []

        self.mocks["select_context"].side_effect = fake_select
        result = self.service.ask("问题", self.session)
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["context_count"], 0)
        self.assertTrue(result["context_incomplete"])

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
        # B5：服务与调试入口共用 open_index，失败后不缓存，下次可重试。
        service = q.QAService()
        calls = []

        def fake_open(collection_name):
            calls.append(collection_name)
            if len(calls) == 1:
                raise RuntimeError("temporary")
            return Mock(), {"parent": self.doc}

        with patch.object(q, "open_index", side_effect=fake_open):
            with self.assertRaises(RuntimeError):
                service._get_resources()
            self.assertIsNone(service._resources)
            first = service._get_resources()
            self.assertIs(first, service._get_resources())
            self.assertEqual(len(calls), 2)


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
