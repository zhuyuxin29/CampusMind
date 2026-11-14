import unittest
from unittest.mock import Mock, patch

from campusmind import cli


class CliTests(unittest.TestCase):
    def test_eof_and_interrupt_exit_cleanly(self):
        for exc in (EOFError, KeyboardInterrupt):
            with self.subTest(exc=exc), patch("builtins.input", side_effect=exc):
                cli.run("query")

    def test_invalid_stage_rejected(self):
        with self.assertRaises(ValueError):
            cli.run("nope")

    def test_answer_stage_uses_qa_service(self):
        fake_service = Mock()
        fake_service.ask.return_value = {
            "status": "insufficient",
            "answer": "当前资料不足以回答这个问题，请补充具体事项或相关资料。",
            "sources": [],
            "insufficient_reason": "model_declined",
            "context_incomplete": False,
            "timings_ms": {"total": 1.0},
        }
        with patch.object(cli, "QAService", return_value=fake_service) as service_class, \
             patch("builtins.input", side_effect=["奖学金如何申请？", "exit"]):
            cli.run("answer")
        service_class.assert_called_once()
        fake_service.ask.assert_called_once()

    def test_query_stage_prints_queries(self):
        with patch("builtins.input", side_effect=["缓考需要什么材料？", "exit"]), \
             patch.object(cli, "_resolve", return_value=("缓考需要什么材料？", ["缓考需要什么材料？"])) as resolve:
            cli.run("query")
        resolve.assert_called_once()


if __name__ == "__main__":
    unittest.main()
