import unittest
from campusmind.session_manager import ChatSession


class SessionTests(unittest.TestCase):
    def test_recent_rounds_and_clear(self):
        session = ChatSession(history_rounds=2)
        for n in range(3):
            session.add_turn(str(n), '回答')
        self.assertEqual([t['question'] for t in session.get_recent_history()], ['1', '2'])
        self.assertEqual(len(session.get_messages()), 6)
        session.clear()
        self.assertEqual(session.get_messages(), [])

    def test_copy_and_isolation(self):
        session, other = ChatSession(), ChatSession()
        sources = [{'page_number': 1}]
        session.add_turn('问题', '回答', sources)
        sources[0]['page_number'] = 9
        history = session.get_recent_history()
        history[0]['sources'][0]['page_number'] = 8
        self.assertEqual(session.get_recent_history()[0]['sources'][0]['page_number'], 1)
        self.assertEqual(other.get_messages(), [])

    def test_invalid_turn_not_saved(self):
        session = ChatSession()
        with self.assertRaises(ValueError):
            session.add_turn(' ', '回答')
        self.assertEqual(session.get_messages(), [])
