"""Tests for parsing complete questions out of a quiz JSON stream in progress."""

import json
import unittest

from tutor_ai import TutorAI


def _q(n, correct_first=True):
    opts = [f"Answer {n}a", f"Answer {n}b", f"Answer {n}c", f"Answer {n}d"]
    return {
        "question": f"Question number {n}?",
        "options": opts,
        "correct_answer": opts[0] if correct_first else opts[2],
        "explanation": f"Because of reason {n}.",
    }


class QuizPartialParseTests(unittest.TestCase):
    def test_nothing_before_the_first_complete_question(self):
        self.assertEqual(TutorAI.parse_quiz_partial(""), [])
        self.assertEqual(TutorAI.parse_quiz_partial('{"questions": [{"question": "Half'), [])

    def test_complete_questions_are_returned_as_the_stream_grows(self):
        full = json.dumps({"questions": [_q(1), _q(2), _q(3)]})
        # cut inside the third question
        cut = full.index('"Question number 3?"') + 5
        partial = TutorAI.parse_quiz_partial(full[:cut])
        self.assertEqual([q["question"] for q in partial], ["Question number 1?", "Question number 2?"])
        whole = TutorAI.parse_quiz_partial(full)
        self.assertEqual(len(whole), 3)

    def test_option_order_is_stable_between_partial_and_final_parse(self):
        full = json.dumps({"questions": [_q(1, correct_first=False), _q(2)]})
        cut = full.index('"Question number 2?"')
        first_partial = TutorAI.parse_quiz_partial(full[:cut])[0]
        first_final = TutorAI._parse_and_validate_quiz(full)[0]
        self.assertEqual(first_partial["options"], first_final["options"])
        self.assertIn(first_final["correct_answer"], first_final["options"])

    def test_braces_and_quotes_inside_strings_do_not_confuse_the_scanner(self):
        q = _q(1)
        q["explanation"] = 'Uses a brace } and a quote \\" inside text'
        full = json.dumps({"questions": [q, _q(2)]})
        cut = full.index('"Question number 2?"')
        partial = TutorAI.parse_quiz_partial(full[:cut])
        self.assertEqual(len(partial), 1)
        self.assertIn("brace }", partial[0]["explanation"])

    def test_malformed_question_is_skipped(self):
        bad = {"question": "No options?", "correct_answer": "x", "explanation": "y"}
        full = json.dumps({"questions": [bad, _q(2)]})
        self.assertEqual([q["question"] for q in TutorAI.parse_quiz_partial(full)], ["Question number 2?"])


if __name__ == "__main__":
    unittest.main()
