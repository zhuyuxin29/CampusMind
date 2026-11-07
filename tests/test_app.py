# 离线界面测试：python -m unittest test_app；不发送模型请求。
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import config
from campusmind import qa_service


class AppTests(unittest.TestCase):
    def make_app(self):
        return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20)

    def test_missing_configuration(self):
        with patch.object(config, "missing_api_keys", return_value=["DEEPSEEK_API_KEY"]):
            app = self.make_app().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.chat_input[0].disabled)
            self.assertTrue(app.warning)

    def test_context_incomplete_caption(self):
        # P1-3：上下文被预算裁剪时，界面要给出提示。
        result = {"status": "answered", "answer": "测试回答[1]", "sources": [],
                  "context_incomplete": True}
        with patch.object(config, "missing_api_keys", return_value=[]), \
             patch.object(qa_service, "ask", return_value=result):
            app = self.make_app().run()
            app.chat_input[0].set_value("测试问题").run()
            self.assertFalse(app.exception)
            self.assertTrue(any("上下文长度预算" in item.value for item in app.caption))

    def test_chat_sources_rerun_and_clear(self):
        def fake_ask(question, session):
            sources = [{"citation_id": 1, "file_name": "学生手册.pdf", "page_start": 2,
                        "page_end": 3, "section_title": "测试章节", "content": "测试原文"}]
            session.add_turn(question, "测试回答[1]", sources)
            return {"status": "answered", "answer": "测试回答[1]", "sources": sources,
                    "timings_ms": {"generate": 30, "total": 40}}
        with patch.object(config, "missing_api_keys", return_value=[]), patch.object(qa_service, "ask", side_effect=fake_ask) as ask:
            app = self.make_app().run()
            app.chat_input[0].set_value("测试问题").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.chat_message), 3)
            self.assertIn("PDF 第 2–3 页", app.expander[0].label)
            app.run()
            self.assertEqual(ask.call_count, 1)
            self.assertEqual(len(app.session_state.chat_session.get_recent_history()), 1)
            app.button(key="new_chat").click().run()
            app.button(key="confirm_clear").click().run()
            self.assertEqual(len(app.chat_message), 1)
            self.assertEqual(app.session_state.chat_session.get_recent_history(), [])

    def test_clarification_and_error(self):
        for result in ({"status": "needs_clarification", "answer": "哪种奖学金？", "sources": []},
                       {"status": "error", "message": "知识库尚未就绪，请先完成资料入库。", "sources": []}):
            with patch.object(config, "missing_api_keys", return_value=[]), patch.object(qa_service, "ask", return_value=result):
                app = self.make_app().run()
                app.chat_input[0].set_value("测试问题").run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state.chat_session.get_recent_history(), [])
                self.assertEqual(len(app.session_state.display_turns), 1)

    def test_session_isolation(self):
        with patch.object(config, "missing_api_keys", return_value=[]):
            one, two = self.make_app().run(), self.make_app().run()
            one.session_state.chat_session.add_turn("甲", "乙")
            self.assertEqual(two.session_state.chat_session.get_recent_history(), [])

    def test_welcome_and_example_only_prefills(self):
        with patch.object(config, "missing_api_keys", return_value=[]), patch.object(qa_service, "ask") as ask:
            app = self.make_app().run()
            self.assertTrue(any("同学，你好呀！我叫团团，是一个校园规章制度问答系统，请问有什么可以帮助你的吗？" in item.value for item in app.markdown))
            self.assertEqual(app.session_state.chat_session.get_recent_history(), [])
            app.button(key="example_0").click().run()
            self.assertFalse(app.exception)
            ask.assert_not_called()

    def test_history_search_and_clear_cancel(self):
        result = {"status": "answered", "answer": "测试回答[1]", "sources": []}
        with patch.object(config, "missing_api_keys", return_value=[]), patch.object(qa_service, "ask", return_value=result) as ask:
            app = self.make_app().run()
            app.chat_input[0].set_value("奖学金问题").run()
            app.chat_input[0].set_value("缓考问题").run()
            app.text_input(key="history_search").set_value("缓考").run()
            links = [item.value for item in app.markdown if 'class="history-link' in item.value]
            self.assertEqual(len(links), 1)
            self.assertIn('href="#turn-1"', links[0])
            self.assertEqual(ask.call_count, 2)
            app.button(key="new_chat").click().run()
            app.button(key="cancel_clear").click().run()
            self.assertEqual(len(app.session_state.display_turns), 2)
            self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()

