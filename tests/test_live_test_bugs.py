"""Regressions for the bugs found by the browser live test of 27 October 2026
(evening): a failed upload wiped the previous document, the first calculation
question stalled while its draft was checked, and messages after a marked
challenge were marked again."""
import asyncio
import io
import os
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

os.environ["DATABASE_URL"] = "sqlite:////tmp/activity-log-tests.sqlite3"
os.environ.setdefault("SESSION_SECRET", "live-test-bugs-secret")

import app as module


class ImmediateThread:
    def __init__(self, target, args=(), **kwargs):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)

    def join(self, timeout=None):
        pass


class UploadKeepsPreviousDocumentTests(unittest.TestCase):
    def test_upload_request_does_not_clear_before_the_file_is_read(self):
        background = MagicMock()
        with ExitStack() as stack:
            clear = stack.enter_context(patch.object(module, "clear_document_rows"))
            stack.enter_context(patch.object(module, "clear_session_data"))
            stack.enter_context(patch.object(module, "process_upload_background", background))
            stack.enter_context(patch.object(module, "create_task"))
            stack.enter_context(patch.object(module.threading, "Thread", ImmediateThread))
            client = module.app.test_client()
            with client.session_transaction() as s:
                s["session_id"] = "keep-session"
                s["pdf_filename"] = "Old notes.pdf"
                s["pdf_hash"] = "old-hash"
            response = client.post("/upload", data={"file": (io.BytesIO(b"not a pdf"), "bad file.pdf")},
                                   content_type="multipart/form-data")
            self.assertEqual(response.status_code, 200)
            clear.assert_not_called()
            module.clear_session_data.assert_not_called()
            with client.session_transaction() as s:
                # Not completed yet: the cookie still names the previous document.
                self.assertEqual(s.get("pdf_filename"), "Old notes.pdf")
                self.assertEqual(s.get("pdf_hash"), "old-hash")
        self.assertEqual(background.call_args.args[4], "bad file.pdf")  # display name

    def test_failed_processing_keeps_state_and_hands_back_the_upload_marker(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "process_document_with_fallback",
                                             return_value=(False, None, "This file does not look like a PDF")))
            clear = stack.enter_context(patch.object(module, "clear_document_rows"))
            restore = stack.enter_context(patch.object(module, "_restore_previous_upload"))
            stack.enter_context(patch.object(module, "update_task_failed"))
            stack.enter_context(patch.object(module, "record_upload"))
            module.process_upload_background("new-task", b"x", "bad.pdf", "keep-session", "bad.pdf", "old-task")
        clear.assert_not_called()
        restore.assert_called_once_with("keep-session", "new-task", "old-task")

    def test_restore_only_when_the_marker_is_still_this_upload(self):
        storage = MagicMock()
        storage.retrieve_content.return_value = "new-task"
        with patch.object(module, "StorageManager", return_value=storage):
            module._restore_previous_upload("s", "new-task", "old-task")
        storage.store_content.assert_called_once_with("s", "current_upload", "old-task")
        storage.reset_mock()
        storage.retrieve_content.return_value = "newer-task"
        with patch.object(module, "StorageManager", return_value=storage):
            module._restore_previous_upload("s", "new-task", "old-task")
        storage.store_content.assert_not_called()

    def test_successful_processing_resets_the_previous_document_first(self):
        order = []
        classifier = MagicMock()
        classifier.detect_document_type_async = AsyncMock(return_value="lecture_notes")
        classifier.is_question_set.return_value = False
        classifier.extract_first_equation_async = AsyncMock(return_value=None)
        classifier.extract_equation_list_async = AsyncMock(return_value=[])
        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "process_document_with_fallback",
                                             return_value=(True, "new notes", None)))
            stack.enter_context(patch.object(module, "clear_document_rows",
                                             side_effect=lambda sid: order.append("clear")))
            stack.enter_context(patch.object(module.primary_storage, "batch_store",
                                             side_effect=lambda ops: order.append([k for _, k, _ in ops])))
            stack.enter_context(patch.object(module.storage_manager, "batch_store"))
            stack.enter_context(patch.object(module.storage_manager, "store_content"))
            stack.enter_context(patch.object(module.storage_manager, "retrieve_content", return_value=None))
            stack.enter_context(patch.object(module, "update_task_complete"))
            stack.enter_context(patch.object(module, "record_upload"))
            stack.enter_context(patch.object(module, "_start_prewarm"))
            stack.enter_context(patch.object(module, "_upload_is_current", return_value=True))
            stack.enter_context(patch.object(module, "_transcribe_equation_pages", return_value=None))
            stack.enter_context(patch.object(module, "TutorAI", return_value=classifier))
            stack.enter_context(patch.object(module.threading, "Thread", ImmediateThread))
            stack.enter_context(patch.object(module, "_generate_summary_early"))
            module.process_upload_background("t", b"x", "n.pdf", "s", "Notes.pdf", "old")
        self.assertEqual(order[:2], ["clear", ["pdf_content", "messages"]])

    def test_document_rows_include_the_practice_flags(self):
        for key in ("calculation_mode_active", "calculation_answer_marked", "messages", "pdf_content",
                    "calculation_prewarm_draft", "calculation_prewarm_served"):
            self.assertIn(key, module.DOCUMENT_STATE_KEYS)


class SupersededUploadTests(unittest.TestCase):
    def test_older_upload_finishing_late_does_not_replace_the_newer_document(self):
        storage = MagicMock()
        storage.retrieve_content.return_value = "newer-task"
        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "process_document_with_fallback",
                                             return_value=(True, "older notes", None)))
            stack.enter_context(patch.object(module, "StorageManager", return_value=storage))
            clear = stack.enter_context(patch.object(module, "clear_document_rows"))
            primary = stack.enter_context(patch.object(module, "primary_storage"))
            failed = stack.enter_context(patch.object(module, "update_task_failed"))
            complete = stack.enter_context(patch.object(module, "update_task_complete"))
            done = stack.enter_context(patch.object(module, "_signal_upload_done"))
            stack.enter_context(patch.object(module, "record_upload"))
            module.process_upload_background("older-task", b"x", "a.pdf", "s", "a.pdf", None)
        clear.assert_not_called()
        primary.batch_store.assert_not_called()
        complete.assert_not_called()
        failed.assert_called_once()
        done.assert_called_once_with("older-task")

    def test_file_chosen_earlier_but_arriving_later_is_turned_away(self):
        stored = {"current_upload_order": 2000}
        storage = MagicMock()
        storage.retrieve_content.side_effect = lambda sid, key: stored.get(key)
        storage.store_content.side_effect = lambda sid, key, value: stored.__setitem__(key, value)
        background = MagicMock()
        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "storage_manager", storage))
            stack.enter_context(patch.object(module, "process_upload_background", background))
            stack.enter_context(patch.object(module, "create_task"))
            stack.enter_context(patch.object(module.threading, "Thread", ImmediateThread))
            client = module.app.test_client()
            with client.session_transaction() as s:
                s["session_id"] = "race-session"
            older = client.post("/upload", data={"file": (io.BytesIO(b"x"), "big.pptx"), "upload_order": "1000"},
                                content_type="multipart/form-data")
            self.assertEqual(older.status_code, 409)
            self.assertTrue(older.get_json()["superseded"])
            background.assert_not_called()
            newer = client.post("/upload", data={"file": (io.BytesIO(b"x"), "small.pdf"), "upload_order": "3000"},
                                content_type="multipart/form-data")
            self.assertEqual(newer.status_code, 200)
            background.assert_called_once()
        self.assertEqual(stored["current_upload_order"], 3000)

    def test_page_ignores_status_of_a_replaced_upload_and_keeps_summary_in_place(self):
        page = open(os.path.join(os.path.dirname(__file__), "..", "templates", "index.html"),
                    encoding="utf-8").read()
        self.assertIn("const isStale = () => uploadSeq !== window.__uploadSeq;", page)
        self.assertEqual(page.count("if (isStale())"), 4)
        self.assertIn("messagesDiv.insertBefore(messageDiv, existingProgressBar);", page)
        self.assertIn("formData.append('upload_order', String(Date.now()));", page)


class PrewarmDraftTests(unittest.TestCase):
    def run_stream(self, responses):
        storage = MagicMock()
        responses = iter(responses)
        storage.batch_retrieve.side_effect = lambda *args: next(responses)
        storage.retrieve_content.return_value = []
        verify = MagicMock(return_value="verify-1")
        consume = MagicMock()
        tutor = MagicMock()
        tutor.close_async_clients = AsyncMock()

        async def live(**kwargs):
            raise AssertionError("no live generation expected")
            yield  # pragma: no cover

        tutor.generate_calculation_question_stream_async = live
        with ExitStack() as stack:
            for name, value in {
                "init_session": lambda: None,
                "get_pdf_content_with_fallback": lambda: "Test notes",
                "record_activity": MagicMock(),
                "StorageManager": lambda: storage,
                "_make_tutor": lambda *args: tutor,
                "consume_prewarmed": consume,
                "start_calculation_verification": verify,
            }.items():
                stack.enter_context(patch.object(module, name, value))
            stack.enter_context(patch.object(module.threading, "Thread", ImmediateThread))
            stack.enter_context(module.app.test_request_context("/calculation_stream", method="POST"))
            module.session["session_id"] = "draft-session"
            with self.assertLogs(level="INFO") as logs:
                response = module.calculation_stream.__wrapped__()
                body = b"".join(response.iter_encoded()).decode()
        return body, "\n".join(logs.output), storage, verify, consume

    def test_finished_draft_is_served_while_it_is_being_checked(self):
        state = {"calc_doc_type": "lecture_notes", "equation_list": ["Eq"], "current_equation_index": 0}
        body, logs, storage, verify, consume = self.run_stream([
            state,
            {"calculation_prewarm_pending": True},
            {"calculation_prewarm_pending": True, "calculation_prewarm_partial": "Draft"},
            {"calculation_prewarm_pending": True, "calculation_prewarm_partial": "Draft question",
             "calculation_prewarm_draft": "Draft question"},
        ])
        self.assertIn('data: "Draft"', body)
        self.assertIn('data: " question"', body)
        self.assertNotIn('"reset"', body)
        self.assertIn("event=prewarm_draft_served", logs)
        storage.store_content.assert_any_call("draft-session", "calculation_prewarm_served", True)
        consume.assert_called_once_with("draft-session", "calculation")
        # The request checks the draft itself; the page swaps in a correction.
        verify.assert_called_once()
        self.assertEqual(verify.call_args.args[1], "Draft question")
        self.assertIn('"verify_task_id": "verify-1"', body)

    def test_prewarm_check_is_dropped_once_the_draft_was_served(self):
        storage = MagicMock()
        storage.retrieve_content.return_value = True
        tutor = MagicMock()
        tutor.verify_calculation_question_async = AsyncMock(return_value={"status": "corrected", "question": "Fixed"})
        finalize = module._checked_calc_prewarm("s", equation="Eq")
        question = "**CHALLENGE** Calculate the value. Please type your numerical answer"
        with patch.object(module, "StorageManager", return_value=storage), \
                patch.object(module.TutorAI, "looks_like_calculation_question", return_value=True):
            self.assertIsNone(asyncio.run(finalize(question, tutor)))
        storage.store_content.assert_not_called()


class TwoStageExamQuestionTests(unittest.TestCase):
    FULL = [{"id": "1a", "question": "WACC"}, {"id": "1b", "question": "Annuity"}, {"id": "2", "question": "FCFF"}]

    def test_matching_id_keeps_paper_order_with_the_first_stage_text(self):
        first = {"id": "Q1(a)", "question": "WACC, stage one wording"}
        merged = module.TutorAI.merge_first_exam_question(first, self.FULL)
        self.assertEqual([q["id"] for q in merged], ["Q1(a)", "1b", "2"])
        self.assertEqual(merged[0]["question"], "WACC, stage one wording")

    def test_later_match_moves_to_the_front_and_loses_nothing(self):
        merged = module.TutorAI.merge_first_exam_question({"id": "1b", "question": "Annuity"}, self.FULL)
        self.assertEqual([q["id"] for q in merged], ["1b", "1a", "2"])

    def test_no_match_or_empty_list(self):
        self.assertEqual(len(module.TutorAI.merge_first_exam_question({"id": "9", "question": "x"}, self.FULL)), 4)
        self.assertEqual(module.TutorAI.merge_first_exam_question({"id": "1a", "question": "x"}, []),
                         [{"id": "1a", "question": "x"}])
        self.assertEqual(module.TutorAI.merge_first_exam_question(None, self.FULL), self.FULL)

    def test_background_stage_stores_the_merged_list_and_clears_pending(self):
        storage = MagicMock()
        tutor = MagicMock()
        tutor.extract_exam_questions_async = AsyncMock(return_value=self.FULL)
        with patch.object(module, "StorageManager", return_value=storage), \
                patch.object(module, "TutorAI", wraps=module.TutorAI) as tutor_cls, \
                patch.object(module, "_upload_is_current", return_value=True):
            tutor_cls.return_value = tutor
            module._extract_full_exam_questions_background("s", "text", "exam_paper", "t", {"id": "1a", "question": "W"})
        stored = dict((k, v) for _, k, v in storage.batch_store.call_args.args[0])
        self.assertFalse(stored["exam_questions_pending"])
        self.assertEqual([q["id"] for q in stored["exam_questions"]], ["1a", "1b", "2"])

    def test_background_failure_still_clears_pending(self):
        storage = MagicMock()
        tutor = MagicMock()
        tutor.extract_exam_questions_async = AsyncMock(side_effect=RuntimeError("model down"))
        with patch.object(module, "StorageManager", return_value=storage), \
                patch.object(module, "TutorAI", return_value=tutor):
            module._extract_full_exam_questions_background("s", "text", "exam_paper", "t", {"id": "1a", "question": "W"})
        storage.store_content.assert_called_with("s", "exam_questions_pending", False)

    def test_next_question_waits_for_the_full_exam_list(self):
        storage = MagicMock()
        rows = iter([{"exam_questions": [{"id": "1a"}], "exam_questions_pending": True},
                     {"exam_questions": self.FULL, "exam_questions_pending": False}])
        storage.batch_retrieve.side_effect = lambda *args: next(rows)
        with patch.object(module.time, "sleep"):
            got = module._wait_for_equation_list(storage, "s", [{"id": "1a"}], 1,
                                                 "exam_questions", "exam_questions_pending")
        self.assertEqual(len(got), 3)
        storage.batch_retrieve.assert_called_with("s", ["exam_questions", "exam_questions_pending"])


class EarlyEssayClickTests(unittest.TestCase):
    def test_failed_preparation_is_generated_live_in_the_same_response(self):
        storage = MagicMock()
        rows = iter([
            {"essay_prewarm_pending": True},                                   # route: still preparing
            {"essay_prewarm_pending": True, "essay_prewarm_partial": "Draft"},  # relayed
            {"essay_prewarm_pending": False},                                  # preparation failed
        ])
        storage.batch_retrieve.side_effect = lambda *args: next(rows)
        live = MagicMock(return_value=iter(['data: "Live round"\n\n', "data: [DONE]\n\n"]))
        with ExitStack() as stack:
            for name, value in {
                "init_session": lambda: None,
                "get_pdf_content_with_fallback": lambda: "article text",
                "record_activity": MagicMock(),
                "StorageManager": lambda: storage,
                "_quickaction_live_stream": live,
            }.items():
                stack.enter_context(patch.object(module, name, value))
            stack.enter_context(patch.object(module.time, "sleep"))
            stack.enter_context(module.app.test_request_context(
                "/quickaction_stream", method="POST", json={"action": "essay"}))
            module.session["session_id"] = "essay-session"
            response = module.quickaction_stream.__wrapped__()
            body = b"".join(response.iter_encoded()).decode()
        self.assertNotIn("click Essay questions again", body)
        self.assertLess(body.index('data: "Draft"'), body.index('"reset": true'))
        self.assertLess(body.index('"reset": true'), body.index('data: "Live round"'))
        live.assert_called_once_with("essay", "essay-session", "article text")

    def test_upload_marks_essay_and_quiz_pending_and_clears_unstarted_ones(self):
        writes = module._early_pending_writes("s")
        for feature in ("essay", "quiz"):
            if feature in module.PREWARM_FEATURES:
                self.assertIn(("s", f"{feature}_prewarm_pending", True), writes)
        storage = MagicMock()
        with patch.object(module, "StorageManager", return_value=storage):
            module._clear_unstarted_pending("s", {"essay"})
        cleared = storage.batch_store.call_args.args[0] if storage.batch_store.called else []
        self.assertNotIn(("s", "essay_prewarm_pending", False), cleared)
        if "quiz" in module.PREWARM_FEATURES:
            self.assertIn(("s", "quiz_prewarm_pending", False), cleared)

    def test_start_prewarm_reports_whether_it_started(self):
        with patch.object(module, "StorageManager"), patch.object(module.threading, "Thread"):
            self.assertFalse(module._start_prewarm("not-a-feature", "s", "t", None, "task", lambda t: None))
            self.assertTrue(module._start_prewarm("not-a-feature", "s", "t", None, "task", lambda t: None, force=True))


class MarkedChallengeTests(unittest.TestCase):
    def test_solution_detection(self):
        self.assertTrue(module._shows_calculation_solution(
            "**Step-by-Step Solution:**\n...\n**Final Answer:** 9.02%\n**Feedback:** Correct."))
        self.assertTrue(module._shows_calculation_solution("Final Answer: -0.48%"))
        self.assertFalse(module._shows_calculation_solution(
            "I can only help with your finance studies. Hint: add the coupon first. Please type your answer."))
        self.assertFalse(module._shows_calculation_solution(""))

    def run_chat(self, marked, message):
        storage = MagicMock()
        storage.batch_retrieve.return_value = {
            "calculation_mode_active": True, "current_calculation_question": "Q", "calculation_answer_marked": marked}
        storage.retrieve_content.return_value = []
        tutor = MagicMock()
        tutor.close_async_clients = AsyncMock()
        tutor.conversation_history = []

        async def chat(msg):
            yield "chat reply"

        async def mark(*args, **kwargs):
            yield "**Final Answer:** 5%"

        tutor.get_response_stream_async = chat
        tutor.check_calculation_answer_stream_async = MagicMock(side_effect=mark)
        with ExitStack() as stack:
            for name, value in {
                "get_pdf_content_with_fallback": lambda: "notes",
                "record_activity": MagicMock(),
                "StorageManager": lambda: storage,
                "_make_tutor": lambda *args: tutor,
                "_settle_calculation_verification": lambda sid: ("Q", None),
                "_current_calc_equation": lambda sid: None,
            }.items():
                stack.enter_context(patch.object(module, name, value))
            stack.enter_context(patch.object(module.threading, "Thread", ImmediateThread))
            stack.enter_context(module.app.test_request_context(
                "/simple_chat_stream", method="POST", json={"message": message}))
            module.session["session_id"] = "chat-session"
            response = module.simple_chat_stream.__wrapped__()
            body = b"".join(response.iter_encoded()).decode()
        return body, tutor, storage

    def test_message_after_a_marked_challenge_goes_to_chat(self):
        body, tutor, _ = self.run_chat(True, "Write me a poem about Belfast")
        tutor.check_calculation_answer_stream_async.assert_not_called()
        self.assertIn("chat reply", body)

    def test_answer_to_an_open_challenge_is_marked_and_flagged(self):
        body, tutor, storage = self.run_chat(False, "5%")
        tutor.check_calculation_answer_stream_async.assert_called_once()
        self.assertIn("Final Answer", body)
        writes = [w for call in storage.batch_store.call_args_list for w in call.args[0]]
        self.assertIn(("chat-session", "calculation_answer_marked", True), writes)


if __name__ == "__main__":
    unittest.main()
