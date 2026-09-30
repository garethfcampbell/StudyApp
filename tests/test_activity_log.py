"""Regression tests for admin activity-log access and event recording."""

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Configure an isolated database before importing the application, whose module
# initialization creates its tables.
os.environ["DATABASE_URL"] = "sqlite:////tmp/activity-log-tests.sqlite3"
os.environ["SESSION_SECRET"] = "activity-log-test-session-secret"

from app import (
    app,
    postgres_db,
    process_upload_background,
    record_activity,
    record_upload,
)  # noqa: E402
from models import ActivityLog  # noqa: E402


class ActivityLogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config.update(TESTING=True)
        with app.app_context():
            postgres_db.drop_all()
            postgres_db.create_all()

    def setUp(self):
        self.client = app.test_client()
        with app.app_context():
            ActivityLog.query.delete()
            postgres_db.session.commit()

    def tearDown(self):
        with app.app_context():
            postgres_db.session.remove()

    def test_admin_routes_deny_missing_and_invalid_credentials(self):
        record_activity(
            "upload",
            session_id="private-session",
            filename="private-record.pdf",
            doc_type="lecture_notes",
        )
        paths = (
            "/admin/activity",
            "/admin/activity?format=csv",
            "/admin/uploads?event=upload",
        )
        credentials = {
            "ADMIN_USER": "module-admin",
            "ADMIN_PASSWORD": "test-password",
            "ADMIN_TOKEN": "test-admin-token",
        }

        with patch.dict(os.environ, credentials):
            for path in paths:
                with self.subTest(path=path, credentials="missing"):
                    response = self.client.get(path)
                    self.assertEqual(response.status_code, 401)
                    self.assertNotIn(b"private-record.pdf", response.data)
                with self.subTest(path=path, credentials="invalid basic"):
                    response = self.client.get(path, auth=("intruder", "wrong"))
                    self.assertEqual(response.status_code, 401)
                    self.assertNotIn(b"private-record.pdf", response.data)

            response = self.client.get(
                "/admin/activity",
                headers={"X-Admin-Token": "invalid-admin-token"},
            )
            self.assertEqual(response.status_code, 401)
            self.assertNotIn(b"private-record.pdf", response.data)

    def test_admin_activity_is_hidden_when_no_credentials_are_configured(self):
        with patch.dict(
            os.environ,
            {"ADMIN_PASSWORD": "", "ADMIN_TOKEN": ""},
        ):
            response = self.client.get("/admin/activity")
        self.assertEqual(response.status_code, 404)
        self.assertNotIn(b"private-record.pdf", response.data)

    def test_valid_basic_auth_and_admin_token_return_the_activity_log(self):
        record_activity(
            "executive_summary",
            session_id="session-123456789",
            filename="audit-document.pdf",
            doc_type="lecture_notes",
        )

        credentials = {
            "ADMIN_USER": "module-admin",
            "ADMIN_PASSWORD": "test-password",
            "ADMIN_TOKEN": "test-admin-token",
        }
        with patch.dict(os.environ, credentials):
            basic_response = self.client.get(
                "/admin/activity",
                auth=("module-admin", "test-password"),
            )
            token_response = self.client.get(
                "/admin/activity",
                headers={"X-Admin-Token": "test-admin-token"},
            )

        for response in (basic_response, token_response):
            with self.subTest(status=response.status_code):
                self.assertEqual(response.status_code, 200)
                self.assertIn(b"Activity log", response.data)
                self.assertIn(b"Executive summary", response.data)
                self.assertIn(b"audit-document.pdf", response.data)

    def test_upload_success_failure_and_feature_use_are_persisted(self):
        record_upload(
            "successful-notes.pdf",
            "upload-session-123456",
            True,
            doc_type="lecture_notes",
            content_chars=321,
        )
        record_upload(
            "failed-paper.pdf",
            "upload-session-654321",
            False,
            error="Could not extract document text",
        )
        record_activity(
            "executive_summary",
            session_id="feature-session-123456",
            filename="successful-notes.pdf",
            doc_type="lecture_notes",
        )
        record_activity(
            "multiple_choice_quiz",
            session_id="feature-session-123456",
            filename="successful-notes.pdf",
            doc_type="lecture_notes",
        )

        with app.app_context():
            rows = ActivityLog.query.order_by(ActivityLog.id.asc()).all()

        self.assertEqual(
            [row.event for row in rows],
            ["upload", "upload", "executive_summary", "multiple_choice_quiz"],
        )
        self.assertTrue(rows[0].success)
        self.assertEqual(rows[0].content_chars, 321)
        self.assertEqual(rows[0].doc_type, "lecture_notes")
        self.assertFalse(rows[1].success)
        self.assertEqual(rows[1].error, "Could not extract document text")
        self.assertTrue(rows[2].success)
        self.assertTrue(rows[3].success)

    def test_upload_processing_records_success_and_failure_outcomes(self):
        classifier = MagicMock()
        classifier.detect_document_type_async = AsyncMock(return_value="lecture_notes")

        with (
            patch(
                "app.process_document_with_fallback",
                return_value=(True, "extracted document text", None),
            ),
            patch("app.storage_manager.store_content"),
            patch("app.primary_storage.store_content"),
            patch("app.TutorAI", return_value=classifier),
            patch("app.update_task_complete"),
        ):
            process_upload_background(
                "successful-task",
                b"document bytes",
                "upload-success.pdf",
                "upload-session",
            )

        with (
            patch(
                "app.process_document_with_fallback",
                return_value=(False, None, "Unsupported document"),
            ),
            patch("app.update_task_failed"),
        ):
            process_upload_background(
                "failed-task",
                b"invalid bytes",
                "upload-failure.pdf",
                "upload-session",
            )

        with app.app_context():
            rows = ActivityLog.query.filter_by(event="upload").order_by(ActivityLog.id).all()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].filename, "upload-success.pdf")
        self.assertTrue(rows[0].success)
        self.assertEqual(rows[0].doc_type, "lecture_notes")
        self.assertEqual(rows[0].content_chars, len("extracted document text"))
        self.assertEqual(rows[1].filename, "upload-failure.pdf")
        self.assertFalse(rows[1].success)
        self.assertEqual(rows[1].error, "Unsupported document")

    def test_quiz_generation_route_records_feature_use(self):
        with (
            patch("app.init_session"),
            patch("app.get_pdf_content_with_fallback", return_value="study notes"),
            patch("app.create_task"),
            patch("app.get_cached_ai_result", return_value=[{"question": "sample"}]),
            patch("app.update_task_complete"),
        ):
            response = self.client.post("/start_quiz_generation")

        self.assertEqual(response.status_code, 202)
        with app.app_context():
            row = ActivityLog.query.filter_by(event="multiple_choice_quiz").one()
        self.assertTrue(row.success)


if __name__ == "__main__":
    unittest.main()