import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from evaluation.run_evaluation import load_dataset, score_result, run_cases
from campusmind.session_manager import ChatSession


class EvaluationTests(unittest.TestCase):
    def test_coverage_not_answer_grade(self):
        case = {'evidence': [{'parent_id': 'a'}, {'parent_id': 'b'}], 'expected_status': 'answered'}
        metrics = score_result(case, {'status': 'answered', 'sources': [{'metadata': {'parent_id': 'a'}}]})
        self.assertEqual(metrics['reference_parent_coverage'], .5)
        self.assertIsNone(metrics['manual_review']['correctness'])

    def test_runs_isolated_and_writes_review(self):
        sessions = []
        def ask(question, session):
            sessions.append(session)
            self.assertEqual(session.get_recent_history(), [])
            session.add_turn(question, '资料不足')
            return {'status': 'insufficient', 'answer': '资料不足', 'sources': [], 'timings_ms': {'total': 1}}
        case = {'id': 'a', 'question': '问题', 'history': [], 'evidence': [], 'answer_points': ['说明不足'],
                'expected_status': 'insufficient', 'human_verified': False}
        with tempfile.TemporaryDirectory() as directory:
            output = run_cases({}, [case, {**case, 'id': 'b'}], Path(directory), ask, ChatSession, {})
            self.assertEqual(len((output / 'results.jsonl').read_text(encoding='utf-8').splitlines()), 2)
            self.assertEqual(json.loads((output / 'summary.json').read_text(encoding='utf-8'))['status_match_rate'], 1)
            self.assertTrue((output / 'review.csv').exists())
        self.assertIsNot(sessions[0], sessions[1])

    def test_duplicate_ids_rejected(self):
        case = {'id': 'x', 'question': '问题', 'history': [], 'evidence': [], 'answer_points': [],
                'expected_status': 'insufficient', 'human_verified': False}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'questions.json'
            path.write_text(json.dumps({'schema_version': 1, 'cases': [case, case]}), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_dataset(path)

    def test_api_exception_keeps_result_without_secret(self):
        case = {'id': 'x', 'question': '问题', 'history': [], 'evidence': [], 'answer_points': [],
                'expected_status': 'insufficient', 'human_verified': False}
        with tempfile.TemporaryDirectory() as folder:
            path = run_cases({}, [case], Path(folder), Mock(side_effect=RuntimeError('secret')), ChatSession, {})
            text = (path / 'results.jsonl').read_text(encoding='utf-8')
            self.assertNotIn('secret', text)
            self.assertEqual(json.loads(text)['result']['status'], 'error')
