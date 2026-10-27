"""Calculation stream lifecycle, empty output, and pending-prewarm regressions."""
import os
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

os.environ["DATABASE_URL"] = "sqlite:////tmp/activity-log-tests.sqlite3"
os.environ.setdefault("SESSION_SECRET", "calculation-stream-test-secret")

import app as module


class ImmediateThread:
    def __init__(self, target, **kwargs):
        self.target = target

    def start(self):
        self.target()


class CalculationStreamTests(unittest.TestCase):
    def run_stream(self, chunks=(), prewarm=None, failure=False, prewarm_partial=None, prewarm_check=None):
        storage = MagicMock()
        state = {
            "calc_doc_type": "lecture_notes",
            "equation_list": ["Example equation"],
            "current_equation_index": 0,
        }
        storage.batch_retrieve.side_effect = (
            lambda sid, keys: {key: state.get(key) for key in keys}
        )
        if prewarm:
            responses = iter([
                state,
                {"calculation_prewarm_pending": True},
                *([{"calculation_prewarm_pending": True, "calculation_prewarm_partial": prewarm_partial}]
                  if prewarm_partial else []),
                {"calculation_prewarm": prewarm},
            ])
            storage.batch_retrieve.side_effect = lambda *args: next(responses)
        storage.retrieve_content.side_effect = (
            lambda sid, key: prewarm_check if key == "calculation_prewarm_check" else [])
        self.verify = MagicMock(return_value=None)
        tutor = MagicMock()
        tutor.close_async_clients = AsyncMock()

        async def produce(**kwargs):
            if failure:
                raise RuntimeError("test model failure")
            for chunk in chunks:
                yield chunk

        tutor.generate_calculation_question_stream_async = produce
        with ExitStack() as stack:
            for name, value in {
                "init_session": lambda: None,
                "get_pdf_content_with_fallback": lambda: "Test notes",
                "record_activity": MagicMock(),
                "StorageManager": lambda: storage,
                "_make_tutor": lambda *args: tutor,
                "_wait_for_equation_list": lambda *args: ["Example equation"],
                "consume_prewarmed": MagicMock(),
                "start_calculation_verification": self.verify,
            }.items():
                stack.enter_context(patch.object(module, name, value))
            stack.enter_context(patch.object(module.threading, "Thread", ImmediateThread))
            stack.enter_context(module.app.test_request_context("/calculation_stream", method="POST"))
            module.session["session_id"] = "test-calculation-session"
            with self.assertLogs(level="INFO") as logs:
                response = module.calculation_stream.__wrapped__()
                body = b"".join(response.iter_encoded()).decode()
            return body, response.headers, "\n".join(logs.output), storage

    def test_empty_output_has_retryable_error_and_correlated_log(self):
        body, headers, logs, storage = self.run_stream()
        self.assertIn('"code": "empty_response"', body)
        self.assertIn('"retryable": true', body)
        self.assertIn("data: [DONE]", body)
        request_id = headers["X-Calculation-Request-ID"]
        self.assertIn("request=" + request_id, logs)
        self.assertIn("event=empty_response", logs)
        self.assertIn("outcome=error text_chars=0", logs)
        storage.batch_store.assert_not_called()

    def test_text_stream_logs_first_text_and_success(self):
        body, _, logs, _ = self.run_stream(["Question ", "text"])
        self.assertIn('data: "Question "', body)
        self.assertIn("event=first_text", logs)
        self.assertIn("outcome=complete text_chars=13", logs)
        self.assertNotIn('"error":', body)

    def test_pending_prewarm_is_served_without_live_generation(self):
        body, _, logs, _ = self.run_stream(prewarm="Cached question", failure=True)
        self.assertIn('data: "Cached question"', body)
        self.assertIn("ready=False pending=True", logs)
        self.assertIn("event=prewarm_served", logs)
        self.assertIn("outcome=complete", logs)

    def test_checked_prewarm_is_not_verified_again_and_keeps_its_answer(self):
        check = {"status": "ok", "answer": "£41,477.60", "reason": "", "draft": "Cached question"}
        body, _, _, storage = self.run_stream(prewarm="Cached question", failure=True, prewarm_check=check)
        self.assertIn('data: "Cached question"', body)
        self.verify.assert_not_called()
        self.assertNotIn("verify_task_id", body)

    def test_relayed_draft_corrected_before_serving_is_reset(self):
        check = {"status": "corrected", "answer": "8,196.16", "reason": "Arithmetic slip", "draft": "Draft question"}
        body, _, _, _ = self.run_stream(prewarm="Corrected question", failure=True,
                                        prewarm_partial="Draft question", prewarm_check=check)
        self.assertLess(body.index('data: "Draft question"'), body.index('"reset": true'))
        self.assertLess(body.index('"reset": true'), body.index('data: "Corrected question"'))
        self.assertNotIn('"corrected"', body)
        self.verify.assert_not_called()

    def test_model_failure_returns_control_error_not_a_fake_question(self):
        body, _, logs, storage = self.run_stream(failure=True)
        self.assertIn('"code": "generation_failed"', body)
        self.assertIn("event=generation_error", logs)
        self.assertIn("outcome=error", logs)
        storage.batch_store.assert_not_called()

    def test_whitespace_only_output_is_treated_as_empty(self):
        body, _, _, _ = self.run_stream(["   "])
        self.assertIn('"code": "empty_response"', body)
