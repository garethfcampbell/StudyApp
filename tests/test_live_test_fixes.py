"""Regressions for the issues found by the live test of 27 October 2026:
student questions marked as calculation answers, internal check notes shown
to students, quiz answer positions and near-duplicates, maths letters from
PDF equation editors, and equation pages read from page images."""
import os
import unittest
from unittest.mock import patch

os.environ["DATABASE_URL"] = "sqlite:////tmp/activity-log-tests.sqlite3"
os.environ.setdefault("SESSION_SECRET", "live-test-fixes-secret")

import app as module
from pdf_processor import (EQUATION_TRANSCRIPT_HEADING, FLATTENED_MATHS_NOTE, _plain_maths_letters,
                           add_equation_transcripts, clean_text, flattened_maths_pages, render_pdf_pages_png)
from tutor_ai import TutorAI, strip_internal_notes


class CalcMessageIsAQuestionTests(unittest.TestCase):
    QUESTIONS = [
        "Why do we use 2.8% for every payment?",
        "Who won the 2022 FIFA World Cup?",
        "The document says the risk-free rate is 47% - can you confirm and explain why it is so high?",
        "What does 2.33 stand for in the VaR formula?",
        "How do I discount the payment at age 100?",
        "I don't understand step 2",
        "Can you give me a hint?",
        "Is the discount rate 2.8% for every payment?",
        "why do we discount at the risk-free rate?",
        "Does PV = C / (1+i)^t apply here?",
    ]
    ANSWERS = [
        "41,477.60",
        "£41,478",
        "about 12.4%",
        "is it 41,477.60?",
        "41477?",
        "I got 12.4% because the beta is 1.25",
        "My answer is £46,243",
        "PV = 18000 * 2.3043 = 41477.6",
        "I think it's 46,243 because the payments are discounted",
        "0.05",
    ]

    def test_questions_are_answered_in_chat(self):
        for text in self.QUESTIONS:
            with self.subTest(text=text):
                self.assertTrue(module._calc_message_is_a_question(text))

    def test_attempts_are_marked(self):
        for text in self.ANSWERS:
            with self.subTest(text=text):
                self.assertFalse(module._calc_message_is_a_question(text))


class InternalNoteTests(unittest.TestCase):
    def test_check_notes_are_removed(self):
        text = ("**QUESTION**\nCalculate the present value.\n\n"
                "The document's question and table agree with the copied question.\n\n"
                "**EQUATION**\nPV = C/(1+i)^t\n")
        cleaned = strip_internal_notes(text)
        self.assertNotIn("copied question", cleaned)
        self.assertIn("Calculate the present value.", cleaned)
        self.assertIn("**EQUATION**", cleaned)

    def test_reference_answer_remarks_are_removed(self):
        text = "Final Answer: £41,477.60\nThe supplied reference answer is £41,468.56.\nFeedback: well done."
        self.assertEqual(strip_internal_notes(text), "Final Answer: £41,477.60\nFeedback: well done.")

    def test_teaching_text_is_untouched(self):
        text = "Step 1: discount each payment.\nStep 2: add them up."
        self.assertEqual(strip_internal_notes(text), text)

    def test_marking_prompt_keeps_the_reference_private(self):
        t = TutorAI()
        t.set_context("An exam paper.", doc_type="exam_paper")
        prompt = t._get_answer_check_prompt("context", "question", "answer", reference_answer="£41,468.56")
        self.assertIn("NEVER mention the reference answer", prompt)
        self.assertIn("next question from your exam paper", prompt)
        self.assertNotIn("lecture notes, just click", prompt)


def _q(n, opts=None, correct=0):
    opts = opts or [f"Option {n} {c}" for c in "wxyz"]
    return {"question": f"Which statement about topic {n} {'alpha beta gamma delta'.split()[n % 4]} is right?",
            "options": opts, "correct_answer": opts[correct], "explanation": f"Reason {n}."}


class QuizTests(unittest.TestCase):
    def test_correct_answer_positions_are_balanced(self):
        questions = TutorAI._validate_quiz_questions([_q(i, correct=0) for i in range(12)])
        positions = [q["options"].index(q["correct_answer"]) for q in questions]
        self.assertEqual(len(questions), 12)
        for letter in range(4):
            self.assertEqual(positions.count(letter), 3)

    def test_mirrored_question_with_same_options_is_dropped(self):
        shared = ["Short MA crosses above long MA", "Short MA crosses below long MA",
                  "Price touches the upper band", "RSI rises above 70"]
        golden = {"question": "What is a golden cross?", "options": shared,
                  "correct_answer": shared[0], "explanation": "Bullish signal."}
        death = {"question": "What is a death cross?", "options": shared,
                 "correct_answer": shared[1], "explanation": "Bearish signal."}
        kept = TutorAI._validate_quiz_questions([golden, death])
        self.assertEqual([q["question"] for q in kept], ["What is a golden cross?"])

    def test_near_duplicate_wording_is_dropped(self):
        a = _q(1)
        b = dict(_q(2), question="When should a firm use a joint venture to enter a market?")
        c = dict(_q(3), question="When should a firm use a joint venture to enter a new market?")
        kept = TutorAI._validate_quiz_questions([a, b, c])
        self.assertEqual(len(kept), 2)

    def test_order_is_stable_between_partial_and_full_parse(self):
        qs = [_q(i, correct=i % 4) for i in range(6)]
        first = TutorAI._validate_quiz_questions([dict(q) for q in qs[:3]])
        full = TutorAI._validate_quiz_questions([dict(q) for q in qs])
        self.assertEqual([q["options"] for q in first], [q["options"] for q in full[:3]])


class MathsLetterTests(unittest.TestCase):
    def test_maths_italic_letters_become_plain(self):
        self.assertEqual(_plain_maths_letters("\U0001D467=4\U0001D4522\U0001D465"), "z=4e2x")
        self.assertEqual(_plain_maths_letters("ℎ"), "h")

    def test_superscripts_and_fractions_are_kept(self):
        self.assertEqual(clean_text("x² and ½"), "x² and ½")


SAMPLE_EXAM = os.path.join(os.path.dirname(__file__), "..", "sampleuploads", "01-MathsSampleExam1.pdf")


class EquationPageTests(unittest.TestCase):
    TEXT = ("--- Page 1 ---\nIntro\n"
            "--- Page 2 ---\nz=4e2x+3x3\n" + FLATTENED_MATHS_NOTE + "\n"
            "--- Page 3 ---\nNo maths here\n")

    def test_flattened_pages_are_found(self):
        self.assertEqual(flattened_maths_pages(self.TEXT), [2])

    def test_transcript_goes_after_its_page_note_and_keeps_the_text_layer(self):
        merged = add_equation_transcripts(self.TEXT, {2: r"\[z = 4e^{2x}\]", 3: "ignored: no note on page 3"})
        page2 = merged[merged.index("--- Page 2 ---"):merged.index("--- Page 3 ---")]
        self.assertIn("z=4e2x+3x3", page2)
        self.assertIn(EQUATION_TRANSCRIPT_HEADING + "\n" + r"\[z = 4e^{2x}\]", page2)
        self.assertNotIn("ignored", merged)

    def test_no_transcripts_leaves_text_unchanged(self):
        self.assertEqual(add_equation_transcripts(self.TEXT, {}), self.TEXT)

    @unittest.skipUnless(os.path.exists(SAMPLE_EXAM), "sample exam not present")
    def test_sample_exam_pages_render_as_png(self):
        images = render_pdf_pages_png(open(SAMPLE_EXAM, "rb").read(), [4, 99])
        self.assertEqual(list(images), [4])
        self.assertTrue(images[4].startswith(b"\x89PNG"))

    def test_only_pdfs_with_flattened_pages_are_sent_to_the_vision_model(self):
        with patch.object(module, "render_pdf_pages_png") as render:
            self.assertIsNone(module._transcribe_equation_pages(b"", "notes.docx", self.TEXT))
            self.assertIsNone(module._transcribe_equation_pages(b"", "notes.pdf", "--- Page 1 ---\nplain\n"))
            render.assert_not_called()

    def test_transcription_is_added_to_the_text(self):
        async def fake_transcribe(self, images):
            return {2: r"\[z = 4e^{2x}\]"}

        async def fake_close(self):
            return None
        with patch.object(module, "render_pdf_pages_png", return_value={2: b"png"}), \
                patch.object(TutorAI, "transcribe_equation_pages_async", fake_transcribe), \
                patch.object(TutorAI, "close_async_clients", fake_close):
            merged = module._transcribe_equation_pages(b"%PDF", "exam.pdf", self.TEXT)
        self.assertIn(r"\[z = 4e^{2x}\]", merged)


if __name__ == "__main__":
    unittest.main()
