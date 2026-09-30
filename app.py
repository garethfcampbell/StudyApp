from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, Response
from flask_compress import Compress
from werkzeug.middleware.proxy_fix import ProxyFix
from markupsafe import escape
from database import postgres_db
from werkzeug.utils import secure_filename
from flask_wtf.csrf import CSRFProtect
import os
import json
import time
import hashlib
import hmac
import logging
import asyncio
import threading
import uuid as uuid_module
from pdf_processor import extract_text_from_file
from tutor_ai import TutorAI
from infographic_email import is_email_configured, normalise_email, email_infographic, email_transport
from database_storage_manager import DatabaseStorageManager as StorageManager
from performance_optimizations import optimized_storage, resource_monitor, rate_limit, start_periodic_cleanup

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
if not app.secret_key:
    raise RuntimeError("SESSION_SECRET environment variable must be set")
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)  # trust one proxy hop for scheme/host/client IP

# Configure PostgreSQL database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
    "pool_size": 20,
    "max_overflow": 40,
}

# Initialize the app with SQLAlchemy
postgres_db.init_app(app)

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Enable gzip compression for all responses
Compress(app)

# CSRF protection will be configured after routes are defined

# Performance optimizations for Autoscale deployment
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 86400  # 24 hours cache for static files


@app.template_global()
def static_version(filename):
    """Cache-busting query value for a static file: its modification time, so a
    deploy that changes the file automatically invalidates the browser cache
    (static files are otherwise cached for 24 hours)."""
    try:
        return int(os.path.getmtime(os.path.join(app.static_folder, filename)))
    except OSError:
        return 0


logging.info(f"INFOGRAPHIC EMAIL: transport = {email_transport() or 'none (set RESEND_API_KEY to enable the email-as-PDF option)'}")
# Secure cookies only in production (HTTPS); allow plain HTTP in local development
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour session timeout
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Security configuration
MAX_MESSAGE_LENGTH = 5000  # Maximum characters for chat messages
MAX_FILENAME_LENGTH = 255  # Maximum filename length

# Initialize storage managers - use optimized version for better performance
storage_manager = StorageManager()  # Keep for backward compatibility
primary_storage = optimized_storage  # Use optimized version as primary

# Start performance monitoring and cleanup
start_periodic_cleanup()

# Configure for production deployment
from deployment_config import configure_for_production, register_health_endpoint
from datetime import datetime

# /health must be available in every environment (dev and production)
register_health_endpoint(app)

if os.environ.get('FLASK_ENV') == 'production':
    app = configure_for_production(app)

# ---------------------------------------------------------------------------
# In-memory TTL cache for deterministic AI generations (per worker process).
# Keyed by (sha256(document_text)[:16], feature) so repeated requests for the
# same document skip the AI call entirely.
# ---------------------------------------------------------------------------
_AI_RESULT_CACHE = {}
_AI_RESULT_CACHE_LOCK = threading.Lock()
AI_RESULT_CACHE_TTL = 3600  # 1 hour
AI_RESULT_CACHE_MAX_ENTRIES = 100

def _ai_cache_key(document_text, feature):
    digest = hashlib.sha256(document_text.encode('utf-8', errors='ignore')).hexdigest()[:16]
    return (digest, feature)

def get_cached_ai_result(document_text, feature):
    """Return a cached AI generation for this document/feature, or None."""
    if not document_text:
        return None
    key = _ai_cache_key(document_text, feature)
    with _AI_RESULT_CACHE_LOCK:
        entry = _AI_RESULT_CACHE.get(key)
        if not entry:
            return None
        value, stored_at = entry
        if time.time() - stored_at > AI_RESULT_CACHE_TTL:
            del _AI_RESULT_CACHE[key]
            return None
        return value

def store_cached_ai_result(document_text, feature, value):
    """Cache a successful AI generation (simple oldest-entry eviction)."""
    if not document_text or not value:
        return
    key = _ai_cache_key(document_text, feature)
    with _AI_RESULT_CACHE_LOCK:
        if key not in _AI_RESULT_CACHE and len(_AI_RESULT_CACHE) >= AI_RESULT_CACHE_MAX_ENTRIES:
            oldest_key = min(_AI_RESULT_CACHE.items(), key=lambda kv: kv[1][1])[0]
            del _AI_RESULT_CACHE[oldest_key]
        _AI_RESULT_CACHE[key] = (value, time.time())

def stream_cached_sse(cached_text, chunk_size=800):
    """Stream previously cached text back over SSE using the same wire format
    as a live AI stream (JSON-encoded data events followed by [DONE])."""
    def generate_cached():
        for i in range(0, len(cached_text), chunk_size):
            yield f"data: {json.dumps(cached_text[i:i + chunk_size])}\n\n"
        yield "data: [DONE]\n\n"
    return Response(generate_cached(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

# Initialize session state helper functions
def init_session():
    """Initialize session variables - optimized for scalability with batch retrieval"""
    resource_monitor.increment('requests_processed')
    
    if 'session_id' not in session:
        session['session_id'] = storage_manager.generate_session_id()
    if 'pdf_filename' not in session:
        session['pdf_filename'] = None
    if 'quiz_active' not in session:
        session['quiz_active'] = False
    if 'equation_active' not in session:
        session['equation_active'] = False
    
    # Use optimized storage for better performance
    session_id = session['session_id']
    
    # OPTIMIZED: Use batch_retrieve to fetch all session data in a SINGLE database query
    # This replaces 7 sequential database calls with 1 batched query
    session_data_types = ['messages', 'quiz_questions', 'current_question_index', 
                         'quiz_score', 'practice_equations', 'current_equation_index', 'equation_score']
    
    # Single batch query instead of 7 sequential queries
    existing_data = primary_storage.batch_retrieve(session_id, session_data_types)
    
    # Only store what doesn't exist (reduce database writes)
    if 'messages' not in existing_data or not existing_data.get('messages'):
        primary_storage.store_content(session_id, 'messages', [])
    
    if 'quiz_questions' not in existing_data or not existing_data.get('quiz_questions'):
        batch_operations = [
            (session_id, 'quiz_questions', []),
            (session_id, 'current_question_index', 0),
            (session_id, 'quiz_score', 0)
        ]
        primary_storage.batch_store(batch_operations)
    
    if 'practice_equations' not in existing_data or not existing_data.get('practice_equations'):
        batch_operations = [
            (session_id, 'practice_equations', []),
            (session_id, 'equation_score', 0)
        ]
        # Only reset the equation index if it has never been set — never overwrite an in-progress value
        if existing_data.get('current_equation_index') is None:
            batch_operations.append((session_id, 'current_equation_index', 0))
        primary_storage.batch_store(batch_operations)

def get_pdf_content():
    """Get PDF content from optimized storage"""
    if 'session_id' not in session:
        return None
    
    content = primary_storage.retrieve_content(session['session_id'], 'pdf_content')
    if content:
        resource_monitor.increment('cache_hits')
    else:
        resource_monitor.increment('cache_misses')
    
    return content

def get_pdf_content_with_fallback():
    """Get PDF content from storage with fallback to both storage systems"""
    if 'session_id' not in session:
        return None
    
    session_id = session['session_id']
    
    # Try optimized storage first
    pdf_content = primary_storage.retrieve_content(session_id, 'pdf_content')
    if pdf_content:
        resource_monitor.increment('cache_hits')
        return pdf_content
    
    # Fallback to original storage manager
    pdf_content = storage_manager.retrieve_content(session_id, 'pdf_content')
    if pdf_content:
        resource_monitor.increment('cache_misses')
        # Cache it in the optimized storage for next time
        primary_storage.store_content(session_id, 'pdf_content', pdf_content)
        return pdf_content
    
    logging.info("PDF content not found in current session")
    return None

def get_tutor_ai():
    """Create a TutorAI instance (returns None if initialization fails)"""
    try:
        return TutorAI()
    except Exception as e:
        logging.error(f"Failed to initialize TutorAI: {e}")
        return None


import re as _re

ESSAY_STATE_KEYS = ('current_essay_question', 'essay_mode_active', 'used_essay_questions')


def _activate_essay_practice(session_id, round_text):
    """After an essay-practice round streams, remember it so the student's next chat
    message is marked as an answer, and record the question so it is not repeated."""
    try:
        storage = StorageManager()
        storage.store_content(session_id, 'current_essay_question', round_text)
        storage.store_content(session_id, 'essay_mode_active', True)
        storage.store_content(session_id, 'calculation_mode_active', False)
        used = storage.retrieve_content(session_id, 'used_essay_questions') or []
        m = _re.search(r"YOUR QUESTION:\s*(.+)", round_text)
        question = m.group(1).strip().strip('*_ ') if m else None
        if not question:
            m2 = _re.search(r"\*\*MODEL QUESTION\*\*\s*\n+\s*\*(.+?)\*", round_text)
            question = m2.group(1).strip() if m2 else None
        if question:
            used.append(question[:300])
        # keep the model question too, so later rounds avoid it as well
        m3 = _re.search(r"\*\*MODEL QUESTION\*\*\s*\n+\s*\*(.+?)\*", round_text)
        if m3:
            used.append(m3.group(1).strip()[:300])
        storage.store_content(session_id, 'used_essay_questions', used[-40:])
    except Exception as e:
        logging.error(f"ESSAY PRACTICE: could not store round state: {e}")


class _PhaseTimer:
    """Accumulates elapsed-time marks for one request and logs them in a single
    line, e.g. "CHAT PHASES: init_session=180ms flags=150ms ... first_token=2900ms"."""
    def __init__(self, label):
        self.label = label
        self.t0 = time.perf_counter()
        self.last = self.t0
        self.marks = []

    def mark(self, name):
        now = time.perf_counter()
        self.marks.append((name, int((now - self.last) * 1000), int((now - self.t0) * 1000)))
        self.last = now

    def log(self):
        parts = " ".join(f"{n}=+{d}ms" for n, d, _ in self.marks)
        total = self.marks[-1][2] if self.marks else 0
        logging.info(f"{self.label} PHASES: {parts} | total_to_last_mark={total}ms")


def _stored_doc_type(session_id=None):
    """Return the document classification ('exam_paper' / 'exercise_set' / 'research_article' / 'lecture_notes') stored
    for the session at upload time, or None if unknown."""
    # Fast path: the type is remembered in the signed session cookie once seen,
    # which is consistent across gunicorn workers and costs no database read.
    try:
        cached = session.get('doc_type')
        if cached:
            return cached
    except Exception:
        pass  # outside a request context
    try:
        sid = session_id or session.get('session_id')
        if sid:
            doc_type = StorageManager().retrieve_content(sid, 'calc_doc_type')
            if doc_type:
                try:
                    session['doc_type'] = doc_type
                except Exception:
                    pass
            return doc_type
    except Exception as e:
        logging.debug(f"Could not read stored document type: {e}")
    return None


def _make_tutor(pdf_content, session_id=None, doc_type=None):
    """Create a TutorAI with the document loaded and its exam/notes type applied,
    so every feature (chat, summary, essay, concepts, quiz, infographic) adapts.
    Background threads with no request context pass doc_type explicitly."""
    tutor_ai = TutorAI()
    tutor_ai.set_context(pdf_content, doc_type=doc_type or _stored_doc_type(session_id))
    return tutor_ai

# Task handling helper functions for PostgreSQL migration
def create_task(task_id, status='pending'):
    """Create a new task in PostgreSQL database"""
    with app.app_context():
        try:
            from models import TaskStatus
            task = TaskStatus(task_id=task_id, status=status)
            postgres_db.session.add(task)
            postgres_db.session.commit()
            logging.debug(f"Created task {task_id} with status {status}")
        except Exception as e:
            postgres_db.session.rollback()
            logging.error(f"Error creating task {task_id}: {e}")
            raise

def update_task_complete(task_id, success=True, data=None, error=None):
    """Update task as complete with results"""
    with app.app_context():
        try:
            from models import TaskStatus
            task = TaskStatus.query.filter_by(task_id=task_id).first()
            if task:
                task.set_complete(success=success, data=data, error=error)
                postgres_db.session.commit()
                logging.debug(f"Updated task {task_id} as complete: success={success}")
            else:
                logging.error(f"Task {task_id} not found for completion update")
        except Exception as e:
            postgres_db.session.rollback()
            logging.error(f"Error updating task {task_id}: {e}")
            raise

def update_task_failed(task_id, error):
    """Update task as failed with error"""
    with app.app_context():
        try:
            from models import TaskStatus
            task = TaskStatus.query.filter_by(task_id=task_id).first()
            if task:
                task.set_failed(error)
                postgres_db.session.commit()
                logging.debug(f"Updated task {task_id} as failed: {error}")
            else:
                logging.error(f"Task {task_id} not found for failure update")
        except Exception as e:
            postgres_db.session.rollback()
            logging.error(f"Error updating task {task_id} failure: {e}")
            raise

def get_task_status(task_id):
    """Get task status and convert to dict for JSON serialization"""
    with app.app_context():
        try:
            from models import TaskStatus
            task = TaskStatus.query.filter_by(task_id=task_id).first()
            if task:
                return task.to_dict()
            return None
        except Exception as e:
            logging.error(f"Error getting task {task_id} status: {e}")
            return None

def cleanup_task(task_id):
    """Clean up completed task (optional cleanup)"""
    with app.app_context():
        try:
            from models import TaskStatus
            task = TaskStatus.query.filter_by(task_id=task_id).first()
            if task:
                postgres_db.session.delete(task)
                postgres_db.session.commit()
                logging.debug(f"Cleaned up task {task_id}")
        except Exception as e:
            postgres_db.session.rollback()
            logging.error(f"Error cleaning up task {task_id}: {e}")

def run_calculation_generation_background(task_id, session_id, pdf_content):
    """
    Background function to generate calculation questions using async gpt-6-sol.
    On first call, extracts an ordered equation list from the notes and stores it in the
    session. Subsequent calls use the stored list and advance the index.
    """
    with app.app_context():
        try:
            logging.info(f"BACKGROUND TASK: Starting calculation generation for task {task_id}")

            tutor_ai = _make_tutor(pdf_content, session_id)

            # Access storage inside app context (before entering async)
            _storage = StorageManager()
            equation_list = _storage.retrieve_content(session_id, 'equation_list')
            exam_questions = _storage.retrieve_content(session_id, 'exam_questions')
            doc_type = _storage.retrieve_content(session_id, 'calc_doc_type')
            current_index = _storage.retrieve_content(session_id, 'current_equation_index') or 0

            async def async_generation():
                nonlocal equation_list, exam_questions, doc_type, current_index
                try:
                    # --- First use: detect document type and extract questions/equations ---
                    if not doc_type:
                        logging.info("BACKGROUND TASK: Detecting document type...")
                        doc_type = await tutor_ai.detect_document_type_async()
                        _storage.store_content(session_id, 'calc_doc_type', doc_type)
                        logging.info(f"BACKGROUND TASK: Document classified as {doc_type}")

                    if doc_type in ("exam_paper", "exercise_set"):
                        # --- EXAM PAPER PATH ---
                        if not exam_questions:
                            logging.info("BACKGROUND TASK: Extracting exam questions...")
                            exam_questions = await tutor_ai.extract_exam_questions_async()
                            logging.info(f"BACKGROUND TASK: Extracted {len(exam_questions) if exam_questions else 0} exam questions")
                            if not exam_questions:
                                return "I couldn't find any calculation questions in this document. Please check it contains numerical questions."
                            _storage.store_content(session_id, 'exam_questions', exam_questions)
                            _storage.store_content(session_id, 'current_equation_index', 0)
                            current_index = 0

                        if current_index >= len(exam_questions):
                            return f"You've worked through all {len(exam_questions)} exam questions — great work! You can upload a new paper to continue practising."

                        current_q = exam_questions[current_index]
                        logging.info(f"BACKGROUND TASK: Generating worked example for exam Q{current_q.get('id', current_index+1)} ({current_index + 1}/{len(exam_questions)})")
                        return await tutor_ai.generate_calculation_question_async(exam_question=current_q)

                    else:
                        # --- LECTURE NOTES PATH (original behaviour) ---
                        if not equation_list:
                            logging.info("BACKGROUND TASK: No equation list found — extracting from notes")
                            equation_list = await tutor_ai.extract_equation_list_async()
                            logging.info(f"BACKGROUND TASK: Extraction returned {len(equation_list) if equation_list else 0} equations")
                            if not equation_list:
                                return "I couldn't find any calculable equations in your notes. Please make sure your document contains mathematical formulas."
                            _storage.store_content(session_id, 'equation_list', equation_list)
                            _storage.store_content(session_id, 'current_equation_index', 0)
                            current_index = 0

                        if current_index >= len(equation_list):
                            return f"You've worked through all {len(equation_list)} equations in your notes — great work! You can upload new notes to continue practising."

                        specific_equation = equation_list[current_index]
                        logging.info(f"BACKGROUND TASK: Generating question for equation {current_index + 1}/{len(equation_list)}: {specific_equation[:80]}")
                        return await tutor_ai.generate_calculation_question_async(specific_equation=specific_equation)
                finally:
                    await tutor_ai.close_async_clients()

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(async_generation())
            loop.close()

            logging.info(f"BACKGROUND TASK: Generation completed for task {task_id}")
            update_task_complete(task_id, success=True, data=result)

        except Exception as e:
            logging.error(f"BACKGROUND TASK: Error in task {task_id}: {e}")
            import traceback
            logging.error(traceback.format_exc())
            update_task_failed(task_id, "An internal error occurred. Please try again.")

def run_calculation_answer_check_background(task_id, challenge_question, user_answer, pdf_content, doc_type=None):
    """
    Background function to check calculation answers using async gpt-6-sol.
    This runs in a separate thread to avoid blocking the web server.
    """
    try:
        logging.info(f"CALC ANSWER BACKGROUND: Starting answer check for task {task_id}")
        
        # Initialize TutorAI and set context
        tutor_ai = _make_tutor(pdf_content, doc_type=doc_type)
        
        # Check calculation answer using async method with gpt-6-sol
        async def async_answer_check():
            try:
                return await tutor_ai.check_calculation_answer_async(challenge_question, user_answer)
            finally:
                await tutor_ai.close_async_clients()
        
        # Run the async function in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_answer_check())
        loop.close()
        
        logging.info(f"CALC ANSWER BACKGROUND: Answer check completed for task {task_id}")
        
        # Store the final result in PostgreSQL Database
        update_task_complete(task_id, success=True, data=result)
        
    except Exception as e:
        logging.error(f"CALC ANSWER BACKGROUND: Error in task {task_id}: {e}")
        update_task_failed(task_id, "An internal error occurred. Please try again.")

def run_summary_generation_background(task_id, pdf_content, doc_type=None):
    """
    Background function to generate executive summaries using async Gemini.
    This runs in a separate thread to avoid blocking the web server.
    """
    try:
        logging.info(f"SUMMARY BACKGROUND: Starting summary generation for task {task_id}")
        
        # Initialize TutorAI and set context
        tutor_ai = _make_tutor(pdf_content, doc_type=doc_type)
        
        # Generate summary using async method
        async def async_summary_generation():
            try:
                return await tutor_ai.generate_cheat_sheet_async()
            finally:
                await tutor_ai.close_async_clients()
        
        # Run the async function in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_summary_generation())
        loop.close()
        
        logging.info(f"SUMMARY BACKGROUND: Generation completed for task {task_id}")

        # Cache the deterministic result and store it in PostgreSQL Database
        store_cached_ai_result(pdf_content, 'summary', result)
        update_task_complete(task_id, success=True, data=result)

    except Exception as e:
        logging.error(f"SUMMARY BACKGROUND: Error in task {task_id}: {e}")
        update_task_failed(task_id, "An internal error occurred. Please try again.")

def run_essay_generation_background(task_id, pdf_content, doc_type=None):
    """
    Background function to generate essay questions using async Gemini.
    This runs in a separate thread to avoid blocking the web server.
    """
    try:
        logging.info(f"ESSAY BACKGROUND: Starting essay generation for task {task_id}")
        
        # Initialize TutorAI and set context
        tutor_ai = _make_tutor(pdf_content, doc_type=doc_type)
        
        # Generate essay using async method
        async def async_essay_generation():
            try:
                return await tutor_ai.generate_essay_question_async()
            finally:
                await tutor_ai.close_async_clients()
        
        # Run the async function in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_essay_generation())
        loop.close()
        
        logging.info(f"ESSAY BACKGROUND: Generation completed for task {task_id}")

        # Cache the deterministic result and store it in PostgreSQL Database
        store_cached_ai_result(pdf_content, 'essay', result)
        update_task_complete(task_id, success=True, data=result)

    except Exception as e:
        logging.error(f"ESSAY BACKGROUND: Error in task {task_id}: {e}")
        update_task_failed(task_id, "An internal error occurred. Please try again.")

def run_key_concepts_generation_background(task_id, pdf_content, doc_type=None):
    """
    Background function to generate key concepts explanations using async Gemini.
    This runs in a separate thread to avoid blocking the web server.
    """
    try:
        logging.info(f"KEY CONCEPTS BACKGROUND: Starting key concepts generation for task {task_id}")
        
        # Initialize TutorAI and set context
        tutor_ai = _make_tutor(pdf_content, doc_type=doc_type)
        
        # Generate key concepts using async method
        async def async_key_concepts_generation():
            try:
                return await tutor_ai.explain_key_concepts_async()
            finally:
                await tutor_ai.close_async_clients()
        
        # Run the async function in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_key_concepts_generation())
        loop.close()
        
        logging.info(f"KEY CONCEPTS BACKGROUND: Generation completed for task {task_id}")

        # Cache the deterministic result and store it in PostgreSQL Database
        store_cached_ai_result(pdf_content, 'key_concepts', result)
        update_task_complete(task_id, success=True, data=result)

    except Exception as e:
        logging.error(f"KEY CONCEPTS BACKGROUND: Error in task {task_id}: {e}")
        update_task_failed(task_id, "An internal error occurred. Please try again.")

# ---- "Email me the infographic as a PDF when it is ready" requests ----
# Stored in the shared PostgreSQL session store keyed by task_id (NOT in process
# memory): the request may be handled by a different gunicorn worker from the
# one running the generation thread, and the student may close the tab. The
# background thread delivers the PDF once the image is complete; if the task has
# already finished, the route sends immediately.
_INFOGRAPHIC_EMAIL_CONTENT_TYPE = 'infographic_email_request'


def register_infographic_email(task_id, email, document_name=None):
    with app.app_context():
        StorageManager().store_content(task_id, _INFOGRAPHIC_EMAIL_CONTENT_TYPE,
                                       {'email': email, 'document_name': document_name})


def pop_infographic_email(task_id):
    """Return {'email', 'document_name'} for the task and remove it, or None."""
    with app.app_context():
        storage = StorageManager()
        req = storage.retrieve_content(task_id, _INFOGRAPHIC_EMAIL_CONTENT_TYPE)
        if req:
            try:
                storage.delete_content(task_id, _INFOGRAPHIC_EMAIL_CONTENT_TYPE)
            except Exception as e:
                logging.debug(f"INFOGRAPHIC EMAIL: could not delete request for {task_id}: {e}")
        return req if isinstance(req, dict) and req.get('email') else None


def _deliver_pending_infographic_email(task_id, image_b64):
    """Send the finished infographic to any address registered for this task."""
    if not image_b64:
        return
    try:
        req = pop_infographic_email(task_id)
    except Exception as e:
        logging.error(f"INFOGRAPHIC EMAIL: could not read pending request for task {task_id}: {e}")
        return
    if not req:
        return
    try:
        logging.info(f"INFOGRAPHIC EMAIL: delivering scheduled PDF for task {task_id} via {email_transport()}")
        email_infographic(req['email'], image_b64, document_name=req.get('document_name'))
        logging.info(f"INFOGRAPHIC EMAIL: delivered PDF for task {task_id}")
        _record_infographic_email_result(task_id, {'status': 'sent', 'email': req['email']})
    except Exception as e:
        logging.error(f"INFOGRAPHIC EMAIL: delivery for task {task_id} failed: {e}")
        _record_infographic_email_result(task_id, {'status': 'failed', 'email': req['email'],
                                                   'error': _email_error_message(e)})


_INFOGRAPHIC_EMAIL_RESULT_TYPE = 'infographic_email_result'


def _record_infographic_email_result(task_id, result):
    try:
        with app.app_context():
            StorageManager().store_content(task_id, _INFOGRAPHIC_EMAIL_RESULT_TYPE, result)
    except Exception as e:
        logging.error(f"INFOGRAPHIC EMAIL: could not record result for task {task_id}: {e}")


def _email_error_message(exc):
    """Student-facing explanation of a send failure, with the provider's reason
    (Resend's messages are short and actionable, e.g. the test-sender restriction)."""
    detail = str(exc).strip()
    if 'only send testing emails' in detail.lower() or 'own email address' in detail.lower():
        return ("The email service is in test mode and can only send to the address that owns "
                "the Resend account. The administrator needs to verify a sending domain and set "
                "RESEND_FROM.")
    if len(detail) > 220:
        detail = detail[:220] + '...'
    return f"Sending the email failed: {detail}" if detail else "Sending the email failed."


def run_infographic_generation_background(task_id, pdf_content, doc_type=None):
    """
    Background function to generate a revision-guide infographic image.
    This runs in a separate thread to avoid blocking the web server.
    """
    try:
        logging.info(f"INFOGRAPHIC BACKGROUND: Starting infographic generation for task {task_id}")

        # Initialize TutorAI and set context
        tutor_ai = _make_tutor(pdf_content, doc_type=doc_type)

        # Generate infographic using async method
        async def async_infographic_generation():
            try:
                return await tutor_ai.generate_infographic_async()
            finally:
                await tutor_ai.close_async_clients()

        # Run the async function in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_infographic_generation())
        loop.close()

        logging.info(f"INFOGRAPHIC BACKGROUND: Generation completed for task {task_id}")

        # Cache so repeat clicks on the same document reuse the image instead
        # of paying for another expensive generation.
        store_cached_ai_result(pdf_content, 'infographic', result)
        update_task_complete(task_id, success=True, data=result)
        _deliver_pending_infographic_email(task_id, result)

    except Exception as e:
        logging.error(f"INFOGRAPHIC BACKGROUND: Error in task {task_id}: {e}")
        try:
            pop_infographic_email(task_id)
        except Exception:
            pass
        update_task_failed(task_id, "An internal error occurred. Please try again.")

def run_chat_response_background(task_id, user_message, pdf_content, conversation_history, doc_type=None):
    """
    Background function to generate chat responses using async processing.
    This runs in a separate thread to avoid blocking the web server.
    """
    try:
        logging.info(f"CHAT BACKGROUND: Starting chat response generation for task {task_id}")
        
        # Initialize TutorAI and set context
        tutor_ai = _make_tutor(pdf_content, doc_type=doc_type)
        tutor_ai.conversation_history = conversation_history or []
        
        # Generate chat response using async method
        async def async_chat_generation():
            try:
                return await tutor_ai.get_response_async(user_message)
            finally:
                await tutor_ai.close_async_clients()
        
        # Run the async function in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_chat_generation())
        loop.close()
        
        logging.info(f"CHAT BACKGROUND: Chat response generation completed for task {task_id}")
        
        # Store the final result in PostgreSQL Database
        update_task_complete(task_id, success=True, data=result)
        
    except Exception as e:
        logging.error(f"CHAT BACKGROUND: Error in task {task_id}: {e}")
        update_task_failed(task_id, "An internal error occurred. Please try again.")

@app.route('/')
def index():
    """Main page - optimized to avoid duplicate database calls"""
    init_session()
    
    # OPTIMIZED: Get message count from cache (already fetched in init_session via batch_retrieve)
    # This avoids a duplicate database call
    session_id = session.get('session_id')
    messages = primary_storage.retrieve_content(session_id, 'messages') or []
    
    return render_template('index.html', 
                         has_document=session.get('pdf_filename') is not None,
                         pdf_filename=session.get('pdf_filename'),
                         infographic_email_enabled=is_email_configured(),
                         infographic_email_prefill=session.get('infographic_email') or '',
                         message_count=len(messages),
                         quiz_active=session.get('quiz_active', False),
                         equation_active=session.get('equation_active', False))

@app.route('/simple_chat', methods=['POST'])
@csrf.exempt
@rate_limit(calls_per_minute=30, use_session=True)  # Session-based rate limiting for chat
def simple_chat():
    """Simple async chat endpoint that calls the AI asynchronously using asyncio"""
    async def async_chat_handler():
        """Async handler for chat processing"""
        tutor_ai = None
        try:
            init_session()

            data = request.get_json()
            if not data or 'message' not in data:
                return {'success': False, 'error': 'No message provided'}, 400
            
            user_message = data['message']
            
            # Enhanced input validation with security logging
            if not isinstance(user_message, str):
                logging.warning(f"SECURITY: Invalid message type from session {session.get('session_id', 'unknown')}: {type(user_message)}")
                resource_monitor.increment('input_validation_failures')
                return {'success': False, 'error': 'Message must be text'}, 400
            
            if len(user_message.strip()) == 0:
                logging.warning(f"SECURITY: Empty message from session {session.get('session_id', 'unknown')}")
                resource_monitor.increment('input_validation_failures')
                return {'success': False, 'error': 'Message cannot be empty'}, 400
                
            if len(user_message) > MAX_MESSAGE_LENGTH:
                logging.warning(f"SECURITY: Message too long from session {session.get('session_id', 'unknown')}: {len(user_message)} characters")
                resource_monitor.increment('input_validation_failures')
                return {'success': False, 'error': f'Message too long (max {MAX_MESSAGE_LENGTH} characters)'}, 400
            
            # Sanitize user input
            user_message = user_message.strip()
            logging.debug("SIMPLE_CHAT: Processing incoming message")
            
            # Check if we're in calculation mode
            session_id = session.get('session_id')
            storage_manager = StorageManager()
            calculation_mode = storage_manager.retrieve_content(session_id, 'calculation_mode_active')
            current_calculation = storage_manager.retrieve_content(session_id, 'current_calculation_question')
            
            logging.info(f"SIMPLE_CHAT: Calculation mode active: {calculation_mode}")
            logging.info(f"SIMPLE_CHAT: Current calculation present: {current_calculation is not None}")
            
            # Handle essay practice mode: the End Practice button (and typed commands)
            essay_mode = storage_manager.retrieve_content(session_id, 'essay_mode_active')
            if essay_mode and user_message.lower().strip() in ['end practice', 'end', 'quit', 'stop']:
                storage_manager.store_content(session_id, 'essay_mode_active', False)
                storage_manager.store_content(session_id, 'current_essay_question', None)
                return {
                    'success': True,
                    'response': 'Essay practice ended. You can start another session anytime! 📝',
                    'end_essay_mode': True
                }, 200

            # Handle calculation mode
            if calculation_mode and current_calculation:
                # Check if user wants to end the session
                if user_message.lower().strip() in ['end practice', 'end', 'quit', 'stop']:
                    storage_manager.store_content(session_id, 'calculation_mode_active', False)
                    storage_manager.store_content(session_id, 'current_calculation_question', None)
                    return {
                        'success': True,
                        'response': 'Practice session ended. You can start a new calculation session anytime! 🎯',
                        'end_calculation_mode': True
                    }, 200
                
                # Check if user wants to skip this question
                elif user_message.lower().strip() in ['skip', 'next']:
                    # Generate a new calculation question
                    # For now, return a message asking to click the button again
                    return {
                        'success': True,
                        'response': 'Question skipped! Click the "Calculation questions" button to get a new practice question. 🔄'
                    }, 200
                
                # Otherwise, treat as an answer to check
                else:
                    logging.info(f"SIMPLE_CHAT: Processing calculation answer: {user_message}")
            
            # Get document context using fallback mechanism
            pdf_content = get_pdf_content_with_fallback()
            
            if not pdf_content:
                logging.info("No document content found for simple chat")
                return {
                    'success': False,
                    'error': 'I need you to upload your lecture notes first before I can help you study! 📚'
                }, 400
            
            # Create tutor AI instance and set context
            tutor_ai = _make_tutor(pdf_content, session_id)
            logging.info(f"SIMPLE_CHAT: Set context with {len(pdf_content)} characters")
            
            # Load existing conversation history from storage to maintain context
            stored_messages = storage_manager.retrieve_content(session_id, 'messages') or []
            for msg in stored_messages:
                tutor_ai.conversation_history.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
            logging.info(f"SIMPLE_CHAT: Loaded {len(stored_messages)} previous messages for context")
            
            # If we're in calculation mode and have a numerical answer, check it
            if calculation_mode and current_calculation:
                # Use calculation answer checking instead of general chat
                try:
                    logging.debug("SIMPLE_CHAT: Calling check_calculation_answer_async")
                    
                    response = await tutor_ai.check_calculation_answer_async(current_calculation, user_message)
                    
                    if response:
                        logging.debug(f"SIMPLE_CHAT: Got calculation answer response of {len(response)} characters")
                        
                        # Store messages in session
                        messages = storage_manager.retrieve_content(session_id, 'messages') or []
                        
                        # Add user answer
                        messages.append({
                            'role': 'user',
                            'content': user_message
                        })
                        
                        # Add assistant response
                        messages.append({
                            'role': 'assistant', 
                            'content': response
                        })
                        
                        # Store updated messages
                        storage_manager.store_content(session_id, 'messages', messages)
                        
                        return {
                            'success': True,
                            'response': response,
                            'calculation_mode': True
                        }, 200
                    else:
                        logging.info("Empty response from calculation answer check")
                        return {
                            'success': False,
                            'error': 'Failed to check your answer'
                        }, 500
                        
                except Exception as e:
                    logging.info(f"Error checking calculation answer: {e}")
                    return {
                        'success': False,
                        'error': 'Error checking your answer. Please try again.'
                    }, 500
            else:
                # Normal chat mode - get response asynchronously using gpt-6-sol primary with Gemini fallback
                response = await tutor_ai.get_response_async(user_message)
                
                if response:
                    logging.info(f"SIMPLE_CHAT: Got response of {len(response)} characters")
                    
                    # Store messages in session
                    messages = storage_manager.retrieve_content(session_id, 'messages') or []
                    
                    # Add user message
                    messages.append({
                        'role': 'user',
                        'content': user_message
                    })
                    
                    # Add assistant response
                    messages.append({
                        'role': 'assistant', 
                        'content': response
                    })
                    
                    # Store updated messages
                    storage_manager.store_content(session_id, 'messages', messages)
                    
                    return {
                        'success': True,
                        'response': response
                    }, 200
                else:
                    logging.error("SIMPLE_CHAT: Empty response from AI")
                    return {
                        'success': False,
                        'error': 'Failed to get response from AI'
                    }, 500
                
        except Exception:
            logging.exception("SIMPLE_CHAT: Error processing chat")
            return {
                'success': False,
                'error': 'An internal error occurred. Please try again.'
            }, 500
        finally:
            if tutor_ai:
                await tutor_ai.close_async_clients()

    # Run async function in event loop
    try:
        result, status_code = asyncio.run(async_chat_handler())
        return jsonify(result), status_code
    except Exception:
        logging.exception("SIMPLE_CHAT: Error in asyncio.run")
        return jsonify({
            'success': False,
            'error': 'An internal error occurred. Please try again.'
        }), 500

@app.route('/simple_chat_stream', methods=['POST'])
@csrf.exempt
@rate_limit(calls_per_minute=30, use_session=True)
def simple_chat_stream():
    """Streaming chat endpoint using Server-Sent Events."""
    import queue

    _ph = _PhaseTimer('CHAT')
    init_session()
    _ph.mark('init_session')

    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'success': False, 'error': 'No message provided'}), 400

    user_message = data['message']
    if not isinstance(user_message, str) or len(user_message.strip()) == 0:
        return jsonify({'success': False, 'error': 'Message must be non-empty text'}), 400
    if len(user_message) > MAX_MESSAGE_LENGTH:
        return jsonify({'success': False, 'error': f'Message too long (max {MAX_MESSAGE_LENGTH} characters)'}), 400

    user_message = user_message.strip()
    session_id = session.get('session_id')
    storage_manager = StorageManager()

    # One query for the practice-mode flags instead of four round trips
    _flags = storage_manager.batch_retrieve(session_id, [
        'calculation_mode_active', 'current_calculation_question',
        'essay_mode_active', 'current_essay_question'])
    calculation_mode = _flags.get('calculation_mode_active')
    current_calculation = _flags.get('current_calculation_question')
    essay_mode = _flags.get('essay_mode_active')
    current_essay = _flags.get('current_essay_question')
    _ph.mark('mode_flags')
    if (essay_mode and current_essay) or (calculation_mode and current_calculation):
        pdf_content = get_pdf_content_with_fallback()
        if not pdf_content:
            return jsonify({'success': False, 'error': 'No document content found.'}), 400

        record_activity('essay_answer' if (essay_mode and current_essay) else 'calculation_answer', session_id)
        tutor_ai = _make_tutor(pdf_content, session_id)

        calc_q = queue.Queue()

        def _answer_check_stream():
            # Essay practice takes precedence: the student's message is their essay answer
            if essay_mode and current_essay:
                return tutor_ai.check_essay_answer_stream_async(current_essay, user_message)
            return tutor_ai.check_calculation_answer_stream_async(current_calculation, user_message)

        def _run_calc_stream():
            with app.app_context():
                async def _consume():
                    full_response = ""
                    try:
                        async for chunk in _answer_check_stream():
                            full_response += chunk
                            calc_q.put(chunk)
                    except Exception as e:
                        logging.error(f"CALC_ANSWER_STREAM: Error: {e}")
                        if not full_response:
                            calc_q.put("I'm having trouble checking your answer right now. Please try again.")
                    finally:
                        try:
                            _storage = StorageManager()
                            msgs = _storage.retrieve_content(session_id, 'messages') or []
                            msgs.append({'role': 'user', 'content': user_message})
                            msgs.append({'role': 'assistant', 'content': full_response})
                            _storage.store_content(session_id, 'messages', msgs)
                        except Exception as e:
                            logging.error(f"CALC_ANSWER_STREAM: Error storing messages: {e}")
                        try:
                            await tutor_ai.close_async_clients()
                        except Exception:
                            pass
                        calc_q.put(None)

                asyncio.run(_consume())

        calc_thread = threading.Thread(target=_run_calc_stream, daemon=True)
        calc_thread.start()

        def generate_calc():
            while True:
                chunk = calc_q.get()
                if chunk is None:
                    yield f"data: [DONE]\n\n"
                    break
                escaped = json.dumps(chunk)
                yield f"data: {escaped}\n\n"

        return Response(generate_calc(), mimetype='text/event-stream',
                        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    pdf_content = get_pdf_content_with_fallback()
    if not pdf_content:
        return jsonify({'success': False, 'error': 'I need you to upload your lecture notes first before I can help you study! 📚'}), 400

    _ph.mark('pdf_content')
    record_activity('chat', session_id)
    tutor_ai = _make_tutor(pdf_content, session_id)
    _ph.mark('tutor')

    stored_messages = storage_manager.retrieve_content(session_id, 'messages') or []
    for msg in stored_messages:
        tutor_ai.conversation_history.append({'role': msg['role'], 'content': msg['content']})
    _ph.mark('history')

    # Use a thread to run the async generator and push chunks into a queue
    q = queue.Queue()

    def _run_stream():
        with app.app_context():
            async def _consume():
                full_response = ""
                first_logged = False
                try:
                    async for chunk in tutor_ai.get_response_stream_async(user_message):
                        if not first_logged:
                            first_logged = True
                            _ph.mark('first_token')
                            _ph.log()
                        full_response += chunk
                        q.put(chunk)
                except Exception as e:
                    logging.error(f"STREAM: Error during streaming: {e}")
                    if not full_response:
                        q.put("I'm having trouble connecting to the AI service right now. Please try again.")
                finally:
                    # Store complete conversation after streaming finishes
                    try:
                        msgs = storage_manager.retrieve_content(session_id, 'messages') or []
                        msgs.append({'role': 'user', 'content': user_message})
                        msgs.append({'role': 'assistant', 'content': full_response})
                        storage_manager.store_content(session_id, 'messages', msgs)
                    except Exception as e:
                        logging.error(f"STREAM: Error storing messages: {e}")
                    try:
                        await tutor_ai.close_async_clients()
                    except Exception:
                        pass
                    q.put(None)  # sentinel

            asyncio.run(_consume())

    thread = threading.Thread(target=_run_stream, daemon=True)
    thread.start()

    def generate():
        while True:
            chunk = q.get()
            if chunk is None:
                # Send final event so the client knows we're done
                yield f"data: [DONE]\n\n"
                break
            # Escape newlines for SSE (each data line must not contain raw newlines)
            escaped = json.dumps(chunk)
            yield f"data: {escaped}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/quickaction_stream', methods=['POST'])
@csrf.exempt
@rate_limit(calls_per_minute=30, use_session=True)
def quickaction_stream():
    """Streaming SSE endpoint for Key Concepts and Essay quick actions."""
    import queue as queue_mod

    init_session()

    data = request.get_json()
    action = (data or {}).get('action', '')
    if action not in ('key_concepts', 'essay'):
        return jsonify({'success': False, 'error': 'Invalid action'}), 400

    session_id = session.get('session_id')
    pdf_content = get_pdf_content_with_fallback()
    if not pdf_content:
        return jsonify({'success': False, 'error': 'No document content found. Please upload lecture notes first.'}), 400

    record_activity('essay_questions' if action == 'essay' else 'key_concepts', session_id)
    # Serve from the AI result cache when this document was already processed
    cached = get_cached_ai_result(pdf_content, action) if action != 'essay' else None
    if cached:
        try:
            _storage = StorageManager()
            msgs = _storage.retrieve_content(session_id, 'messages') or []
            msgs.append({'role': 'user', 'content': 'Explanation of key concepts' if action == 'key_concepts' else 'Essay question'})
            msgs.append({'role': 'assistant', 'content': cached})
            _storage.store_content(session_id, 'messages', msgs)
        except Exception as e:
            logging.error(f"QUICKACTION STREAM: Error storing cached messages: {e}")
        return stream_cached_sse(cached)

    q = queue_mod.Queue()

    def _run_stream():
      with app.app_context():
        async def _consume():
            full_response = ""
            stream_ok = False
            tutor_ai = None
            try:
                tutor_ai = _make_tutor(pdf_content, session_id)

                if action == 'key_concepts':
                    gen = tutor_ai.explain_key_concepts_stream_async()
                else:
                    used_questions = StorageManager().retrieve_content(session_id, 'used_essay_questions') or []
                    gen = tutor_ai.generate_essay_question_stream_async(used_questions=used_questions)

                async for chunk in gen:
                    full_response += chunk
                    q.put(chunk)
                stream_ok = True
            except Exception as e:
                logging.error(f"QUICKACTION STREAM: Error during streaming ({action}): {e}")
                if not full_response:
                    q.put("I'm having trouble right now. Please try again in a moment.")
            finally:
                if stream_ok and full_response and action != 'essay':
                    store_cached_ai_result(pdf_content, action, full_response)
                if action == 'essay' and stream_ok and full_response:
                    _activate_essay_practice(session_id, full_response)
                # Store the response in message history
                try:
                    storage_manager = StorageManager()
                    msgs = storage_manager.retrieve_content(session_id, 'messages') or []
                    msgs.append({'role': 'user', 'content': 'Explanation of key concepts' if action == 'key_concepts' else 'Essay question'})
                    msgs.append({'role': 'assistant', 'content': full_response})
                    storage_manager.store_content(session_id, 'messages', msgs)
                except Exception as e:
                    logging.error(f"QUICKACTION STREAM: Error storing messages: {e}")
                try:
                    if tutor_ai:
                        await tutor_ai.close_async_clients()
                except Exception:
                    pass
                q.put(None)

        asyncio.run(_consume())

    thread = threading.Thread(target=_run_stream, daemon=True)
    thread.start()

    def generate():
        while True:
            chunk = q.get()
            if chunk is None:
                yield f"data: [DONE]\n\n"
                break
            escaped = json.dumps(chunk)
            yield f"data: {escaped}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/calculation_stream', methods=['POST'])
@csrf.exempt
@rate_limit(calls_per_minute=10, use_session=True)
def calculation_stream():
    """Streaming SSE endpoint for calculation question generation (exam & lecture notes)."""
    import queue as queue_mod

    init_session()

    session_id = session.get('session_id')
    pdf_content = get_pdf_content_with_fallback()
    if not pdf_content:
        return jsonify({'success': False, 'error': 'No document content found. Please upload lecture notes first.'}), 400

    record_activity('calculation_question', session_id)
    # Read session data before entering the background thread
    storage_manager = StorageManager()
    doc_type = storage_manager.retrieve_content(session_id, 'calc_doc_type')
    exam_questions = storage_manager.retrieve_content(session_id, 'exam_questions')
    equation_list = storage_manager.retrieve_content(session_id, 'equation_list')
    current_index = storage_manager.retrieve_content(session_id, 'current_equation_index') or 0

    q = queue_mod.Queue()

    def _run_stream():
      with app.app_context():
        async def _consume():
            nonlocal doc_type, exam_questions, equation_list, current_index
            full_response = ""
            try:
                tutor_ai = _make_tutor(pdf_content, session_id)
                _storage = StorageManager()

                # --- Detect document type on first call ---
                if not doc_type:
                    logging.info("CALCULATION STREAM: Detecting document type...")
                    doc_type = await tutor_ai.detect_document_type_async()
                    _storage.store_content(session_id, 'calc_doc_type', doc_type)
                    logging.info(f"CALCULATION STREAM: Document classified as {doc_type}")

                if doc_type in ("exam_paper", "exercise_set"):
                    # --- EXAM PAPER PATH (streaming) ---
                    if not exam_questions:
                        logging.info("CALCULATION STREAM: Extracting exam questions...")
                        exam_questions = await tutor_ai.extract_exam_questions_async()
                        if not exam_questions:
                            q.put("I couldn't find any calculation questions in this document. Please check it contains numerical questions.")
                            return
                        _storage.store_content(session_id, 'exam_questions', exam_questions)
                        _storage.store_content(session_id, 'current_equation_index', 0)
                        current_index = 0

                    if current_index >= len(exam_questions):
                        q.put(f"You've worked through all {len(exam_questions)} exam questions — great work! You can upload a new paper to continue practising.")
                        return

                    current_q = exam_questions[current_index]
                    logging.info(f"CALCULATION STREAM: Streaming worked example for exam Q{current_q.get('id', current_index+1)} ({current_index + 1}/{len(exam_questions)})")
                    context_truncated = tutor_ai._get_truncated_context()
                    async for chunk in tutor_ai._generate_exam_worked_example_stream(context_truncated, current_q):
                        full_response += chunk
                        q.put(chunk)

                else:
                    # --- LECTURE NOTES PATH (streaming) ---
                    if not equation_list:
                        logging.info("CALCULATION STREAM: Extracting equation list from notes...")
                        equation_list = await tutor_ai.extract_equation_list_async()
                        if not equation_list:
                            q.put("I couldn't find any calculable equations in your notes. Please make sure your document contains mathematical formulas.")
                            return
                        _storage.store_content(session_id, 'equation_list', equation_list)
                        _storage.store_content(session_id, 'current_equation_index', 0)
                        current_index = 0

                    if current_index >= len(equation_list):
                        q.put(f"You've worked through all {len(equation_list)} equations in your notes — great work! You can upload new notes to continue practising.")
                        return

                    specific_equation = equation_list[current_index]
                    logging.info(f"CALCULATION STREAM: Streaming question for equation {current_index + 1}/{len(equation_list)}")
                    async for chunk in tutor_ai.generate_calculation_question_stream_async(specific_equation=specific_equation):
                        full_response += chunk
                        q.put(chunk)

            except Exception as e:
                logging.error(f"CALCULATION STREAM: Error: {e}")
                import traceback
                logging.error(traceback.format_exc())
                if not full_response:
                    q.put("I'm having trouble generating a calculation question right now. Please try again in a moment.")
            finally:
                # Store the response for calculation answer checking
                try:
                    _storage = StorageManager()
                    if full_response:
                        _storage.store_content(session_id, 'current_calculation_question', full_response)
                        _storage.store_content(session_id, 'calculation_mode_active', True)
                        _storage.store_content(session_id, 'essay_mode_active', False)
                    msgs = _storage.retrieve_content(session_id, 'messages') or []
                    msgs.append({'role': 'assistant', 'content': full_response})
                    _storage.store_content(session_id, 'messages', msgs)
                except Exception as e:
                    logging.error(f"CALCULATION STREAM: Error storing messages: {e}")
                try:
                    await tutor_ai.close_async_clients()
                except Exception:
                    pass
                q.put(None)

        asyncio.run(_consume())

    thread = threading.Thread(target=_run_stream, daemon=True)
    thread.start()

    def generate():
        while True:
            chunk = q.get()
            if chunk is None:
                yield f"data: [DONE]\n\n"
                break
            escaped = json.dumps(chunk)
            yield f"data: {escaped}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/summary_stream', methods=['POST'])
@csrf.exempt
@rate_limit(calls_per_minute=10, use_session=True)
def summary_stream():
    """Streaming SSE endpoint for executive summary generation."""
    import queue as queue_mod

    init_session()

    session_id = session.get('session_id')

    # Retrieve PDF content (already stored by the time the upload completes)
    pdf_content = get_pdf_content_with_fallback()
    if not pdf_content:
        return jsonify({'success': False, 'error': 'Document content not available. Please try uploading your file again.'}), 400

    record_activity('executive_summary', session_id)
    # The upload job starts the summary as soon as the text is stored: use that
    # result if it is ready (or about to be), otherwise the process cache.
    _storage_s = StorageManager()
    early = _storage_s.retrieve_content(session_id, 'executive_summary')
    if not early and _storage_s.retrieve_content(session_id, 'summary_pending'):
        deadline = time.time() + 12
        while time.time() < deadline:
            time.sleep(0.3)
            early = _storage_s.retrieve_content(session_id, 'executive_summary')
            if early or not _storage_s.retrieve_content(session_id, 'summary_pending'):
                break
    cached = early or get_cached_ai_result(pdf_content, 'summary')
    if cached:
        try:
            _storage = StorageManager()
            msgs = _storage.retrieve_content(session_id, 'messages') or []
            msgs.append({'role': 'assistant', 'content': cached})
            _storage.store_content(session_id, 'messages', msgs)
        except Exception as e:
            logging.error(f"SUMMARY STREAM: Error storing cached messages: {e}")
        return stream_cached_sse(cached)

    q = queue_mod.Queue()

    def _run_stream():
      with app.app_context():
        async def _consume():
            full_response = ""
            stream_ok = False
            tutor_ai = None
            try:
                tutor_ai = _make_tutor(pdf_content, session_id)
                async for chunk in tutor_ai.generate_cheat_sheet_stream_async():
                    full_response += chunk
                    q.put(chunk)
                stream_ok = True
            except Exception as e:
                logging.error(f"SUMMARY STREAM: Error: {e}")
                if not full_response:
                    q.put("I'm having trouble generating a summary right now. Please try again.")
            finally:
                if stream_ok and full_response:
                    store_cached_ai_result(pdf_content, 'summary', full_response)
                try:
                    storage_manager = StorageManager()
                    msgs = storage_manager.retrieve_content(session_id, 'messages') or []
                    msgs.append({'role': 'assistant', 'content': full_response})
                    storage_manager.store_content(session_id, 'messages', msgs)
                except Exception as e:
                    logging.error(f"SUMMARY STREAM: Error storing messages: {e}")
                try:
                    if tutor_ai:
                        await tutor_ai.close_async_clients()
                except Exception:
                    pass
                q.put(None)

        asyncio.run(_consume())

    thread = threading.Thread(target=_run_stream, daemon=True)
    thread.start()

    def generate():
        while True:
            chunk = q.get()
            if chunk is None:
                yield f"data: [DONE]\n\n"
                break
            escaped = json.dumps(chunk)
            yield f"data: {escaped}\n\n"

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@app.route('/start_summary_generation', methods=['POST'])
@csrf.exempt
@rate_limit(calls_per_minute=10, use_session=True)
def start_summary_generation():
    """Start background summary generation using polling pattern"""
    try:
        init_session()

        pdf_content = get_pdf_content_with_fallback()

        if not pdf_content:
            logging.error("No document content found for summary generation")
            # Return 200 with error message instead of 400 to prevent HTTP errors in frontend
            return jsonify({'success': False, 'error': 'Document content not available. Please try uploading your file again.'}), 200

        # Generate unique task ID
        task_id = str(uuid_module.uuid4())

        # Set initial status in PostgreSQL Database
        create_task(task_id, "pending")

        # Serve from the AI result cache when this document was already summarised
        cached = get_cached_ai_result(pdf_content, 'summary')
        if cached:
            update_task_complete(task_id, success=True, data=cached)
            return jsonify({"task_id": task_id}), 202

        # Start background task in separate thread
        thread = threading.Thread(
            target=run_summary_generation_background,
            args=(task_id, pdf_content, _stored_doc_type()),
            daemon=True
        )
        thread.start()

        return jsonify({"task_id": task_id}), 202

    except Exception:
        logging.exception("Critical error in summary generation")
        return jsonify({'error': 'An internal error occurred. Please try again.'}), 500

@app.route('/summary_status/<task_id>', methods=['GET'])
def get_summary_status(task_id):
    """Get the status of a summary generation task"""
    try:
        # Retrieve status from PostgreSQL Database
        task_result = get_task_status(task_id)
        
        if not task_result:
            return jsonify({"status": "not_found"}), 404

        result_dict = task_result

        # If task is complete, clear the summary flag but don't store in messages
        # (the frontend will handle displaying it to avoid duplication)
        if result_dict.get("status") == "complete" and result_dict.get("success"):
            session_id = session.get('session_id')
            if session_id:
                # Clear the summary flag
                session['needs_summary'] = False

                logging.info(f"SUMMARY POLLING: Summary completed for task {task_id}")

                # Clean up the task from database after successful completion
                try:
                    cleanup_task(task_id)
                except:
                    pass

        return jsonify(result_dict)

    except Exception:
        logging.exception("SUMMARY POLLING: Error checking task status")
        return jsonify({"status": "error", "error": "An internal error occurred. Please try again."}), 500

def run_quiz_generation_background(task_id, pdf_content, doc_type=None):
    """Background task to generate retrieval quiz using async methods"""
    def run_async():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async function
            result = loop.run_until_complete(run_quiz_generation_async(task_id, pdf_content, doc_type=doc_type))
            return result
        finally:
            loop.close()
    
    return run_async()

async def run_quiz_generation_async(task_id, pdf_content, doc_type=None):
    """Async worker function for quiz generation"""
    try:
        logging.info(f"ASYNC QUIZ WORKER: Starting async quiz generation for task {task_id}")
        
        # Update status to running (we'll handle this by updating with partial status)
        # Note: PostgreSQL model handles running state differently
        pass  # Running state will be implicit between pending and complete
        
        # Create TutorAI instance and set context
        tutor_ai = _make_tutor(pdf_content, doc_type=doc_type)
        
        # Generate quiz using async method
        try:
            quiz_questions = await tutor_ai.generate_retrieval_quiz_async()
        finally:
            await tutor_ai.close_async_clients()
        
        if quiz_questions:
            logging.info(f"ASYNC QUIZ WORKER: Successfully generated {len(quiz_questions)} questions for task {task_id}")
            # Cache the deterministic result and store it in PostgreSQL Database
            store_cached_ai_result(pdf_content, 'quiz', quiz_questions)
            update_task_complete(task_id, success=True, data=quiz_questions)
        else:
            logging.error(f"ASYNC QUIZ WORKER: No questions generated for task {task_id}")
            update_task_failed(task_id, "No quiz questions could be generated. Please try again.")

    except Exception as e:
        logging.error(f"ASYNC QUIZ WORKER: Error in quiz generation for task {task_id}: {e}")
        update_task_failed(task_id, "Quiz generation failed. Please try again.")

@app.route('/start_quiz_generation', methods=['POST'])
@csrf.exempt
@rate_limit(calls_per_minute=10, use_session=True)
def start_quiz_generation():
    """Start background retrieval quiz generation using polling pattern"""
    try:
        logging.info("QUIZ POLLING: Starting quiz generation")
        init_session()

        # Set context from stored content with fallback
        session_id = session.get('session_id')
        logging.info(f"QUIZ POLLING: Current session_id: {session_id}")

        pdf_content = get_pdf_content_with_fallback()
        logging.info(f"QUIZ POLLING: PDF content retrieved: {pdf_content is not None}")

        if not pdf_content:
            logging.error("QUIZ POLLING: No document content found even with fallback")
            return jsonify({'error': 'No document content found'}), 400

        record_activity('multiple_choice_quiz')

        # Generate unique task ID
        task_id = str(uuid_module.uuid4())

        # Set initial status in PostgreSQL Database
        create_task(task_id, "pending")

        # Serve from the AI result cache when a quiz was already generated for this document
        cached = get_cached_ai_result(pdf_content, 'quiz')
        if cached:
            update_task_complete(task_id, success=True, data=cached)
            return jsonify({"task_id": task_id}), 202

        # Start background task in separate thread
        thread = threading.Thread(
            target=run_quiz_generation_background,
            args=(task_id, pdf_content, _stored_doc_type()),
            daemon=True
        )
        thread.start()

        logging.info(f"QUIZ POLLING: Background task started with ID: {task_id}")

        return jsonify({"task_id": task_id}), 202

    except Exception:
        logging.exception("QUIZ POLLING: Critical error")
        return jsonify({'error': 'An internal error occurred. Please try again.'}), 500

@app.route('/quiz_status/<task_id>', methods=['GET'])
def get_quiz_status(task_id):
    """Get the status of a quiz generation task"""
    try:
        # Retrieve status from PostgreSQL Database
        task_result = get_task_status(task_id)
        
        if not task_result:
            return jsonify({"status": "not_found"}), 404

        result_dict = task_result

        # If task is complete, also update session storage
        if result_dict.get("status") == "complete" and result_dict.get("success"):
            session_id = session.get('session_id')
            if session_id:
                storage_manager = StorageManager()
                quiz_questions = result_dict.get("data", [])
                
                # Store quiz questions in file storage
                storage_manager.store_content(session_id, 'quiz_questions', quiz_questions)
                storage_manager.store_content(session_id, 'current_question_index', 0)
                storage_manager.store_content(session_id, 'quiz_score', 0)
                
                # Quiz questions stored successfully
                
                # Clean up the task from database after successful completion
                try:
                    cleanup_task(task_id)
                except:
                    pass
        
        return jsonify(result_dict)

    except Exception:
        logging.exception("Error checking quiz task status")
        return jsonify({"status": "error", "error": "An internal error occurred. Please try again."}), 500

@app.route('/start_calculation_generation', methods=['POST'])
@csrf.exempt
@rate_limit(calls_per_minute=10, use_session=True)
def start_calculation_generation():
    """Start background calculation question generation using polling pattern"""
    try:
        logging.info("POLLING: Starting calculation generation")
        init_session()
        
        # Parse request data safely
        data = None
        try:
            if request.content_type == 'application/json':
                data = request.get_json(force=True, silent=True)
        except Exception as e:
            logging.info(f"POLLING: No JSON data in request: {e}")
            data = None
        
        # Set context from stored content with fallback
        session_id = session.get('session_id')
        logging.info(f"POLLING: Current session_id: {session_id}")
        
        pdf_content = get_pdf_content_with_fallback()
        logging.info(f"POLLING: PDF content retrieved: {pdf_content is not None}")
        
        if not pdf_content:
            logging.error("POLLING: No document content found even with fallback")
            return jsonify({'error': 'No document content found'}), 400

        record_activity('calculation_question')
        
        # Generate unique task ID
        task_id = str(uuid_module.uuid4())
        
        # Set initial status in PostgreSQL Database
        create_task(task_id, "pending")
        
        # Start background task in separate thread
        thread = threading.Thread(
            target=run_calculation_generation_background,
            args=(task_id, session_id, pdf_content),
            daemon=True
        )
        thread.start()

        logging.info(f"POLLING: Background task started with ID: {task_id}")

        return jsonify({"task_id": task_id}), 202

    except Exception:
        logging.exception("POLLING: Critical error")
        return jsonify({'error': 'An internal error occurred. Please try again.'}), 500

@app.route('/calculation_status/<task_id>', methods=['GET'])
def get_calculation_status(task_id):
    """Get the status of a calculation generation task"""
    try:
        # Retrieve status from PostgreSQL Database
        task_result = get_task_status(task_id)
        
        if not task_result:
            return jsonify({"status": "not_found"}), 404

        result_dict = task_result

        # If task is complete, update session storage
        if result_dict.get("status") == "complete" and result_dict.get("success"):
            session_id = session.get('session_id')
            if session_id:
                storage_manager = StorageManager()
                question_text = result_dict.get("data", "")

                if question_text:
                    storage_manager.store_content(session_id, 'current_calculation_question', question_text)
                    storage_manager.store_content(session_id, 'calculation_mode_active', True)

                try:
                    cleanup_task(task_id)
                except:
                    pass

        return jsonify(result_dict)

    except Exception:
        logging.exception("Error checking calculation task status")
        return jsonify({"status": "error", "error": "An internal error occurred. Please try again."}), 500


@app.route('/increment_equation_index', methods=['POST'])
@csrf.exempt
def increment_equation_index():
    """Advance to the next equation in the session's equation list."""
    try:
        init_session()
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session'}), 400

        storage_manager = StorageManager()
        current_index = storage_manager.retrieve_content(session_id, 'current_equation_index') or 0
        new_index = current_index + 1
        storage_manager.store_content(session_id, 'current_equation_index', new_index)

        equation_list = storage_manager.retrieve_content(session_id, 'equation_list') or []
        total = len(equation_list)
        logging.info(f"INCREMENT: index {current_index} → {new_index} (total {total})")

        return jsonify({'index': new_index, 'total': total})

    except Exception:
        logging.exception("Error incrementing equation index")
        return jsonify({'error': 'An internal error occurred. Please try again.'}), 500

@app.route('/start_calculation_answer_check', methods=['POST'])
@csrf.exempt
@rate_limit(calls_per_minute=10, use_session=True)
def start_calculation_answer_check():
    """Start background calculation answer checking using polling pattern"""
    try:
        init_session()

        # Parse request data — require application/json to prevent CSRF via form submissions
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Request must be JSON (Content-Type: application/json)'}), 415
        challenge_question = data.get('challenge_question')
        user_answer = data.get('user_answer')

        if not challenge_question or not user_answer:
            return jsonify({'error': 'Missing challenge_question or user_answer'}), 400

        # Input validation: both fields must be text with sane length limits
        if not isinstance(challenge_question, str) or not isinstance(user_answer, str):
            resource_monitor.increment('input_validation_failures')
            return jsonify({'error': 'challenge_question and user_answer must be text'}), 400
        if len(user_answer) > MAX_MESSAGE_LENGTH:
            resource_monitor.increment('input_validation_failures')
            return jsonify({'error': f'Answer too long (max {MAX_MESSAGE_LENGTH} characters)'}), 400
        if len(challenge_question) > 50000:
            resource_monitor.increment('input_validation_failures')
            return jsonify({'error': 'Challenge question too long'}), 400

        # Set context from stored content with fallback
        session_id = session.get('session_id')

        pdf_content = get_pdf_content_with_fallback()

        if not pdf_content:
            logging.info("No document content found for calculation answer checking")
            return jsonify({'error': 'No document content found'}), 400

        record_activity('calculation_answer')

        # Generate unique task ID
        task_id = str(uuid_module.uuid4())

        # Set initial status in PostgreSQL Database
        create_task(task_id, "pending")

        # Start background task in separate thread
        thread = threading.Thread(
            target=run_calculation_answer_check_background,
            args=(task_id, challenge_question, user_answer, pdf_content, _stored_doc_type()),
            daemon=True
        )
        thread.start()


        return jsonify({"task_id": task_id}), 202

    except Exception:
        logging.exception("Critical error in calculation answer checking")
        return jsonify({'error': 'An internal error occurred. Please try again.'}), 500

@app.route('/calculation_answer_status/<task_id>', methods=['GET'])
def get_calculation_answer_status(task_id):
    """Get the status of a calculation answer check task"""
    try:
        # Retrieve status from PostgreSQL Database
        task_result = get_task_status(task_id)
        
        if not task_result:
            return jsonify({"status": "not_found"}), 404

        result_dict = task_result

        # If task is complete, also update session storage
        if result_dict.get("status") == "complete" and result_dict.get("success"):
            session_id = session.get('session_id')
            if session_id:
                storage_manager = StorageManager()
                evaluation_response = result_dict.get("data", "")

                # Add evaluation to messages
                messages = storage_manager.retrieve_content(session_id, 'messages') or []
                messages.append({
                    "role": "assistant",
                    "content": evaluation_response
                })
                storage_manager.store_content(session_id, 'messages', messages)

                # Clear any current calculation question
                storage_manager.delete_content(session_id, 'current_calculation_question')

                # Clean up the task from database after successful completion
                try:
                    cleanup_task(task_id)
                except:
                    pass

        return jsonify(result_dict)

    except Exception:
        logging.exception("Error checking calculation answer task status")
        return jsonify({"status": "error", "error": "An internal error occurred. Please try again."}), 500



def process_document_with_fallback(file, max_retries=3):
    """
    Process document with fallback mechanism
    Returns: (success, pdf_text, error_message)
    """
    for attempt in range(max_retries):
        try:
            # Reset file pointer for each attempt
            file.seek(0)
            
            # Extract text from file
            pdf_text = extract_text_from_file(file)
            
            if pdf_text and pdf_text.strip():
                return True, pdf_text, None
            else:
                if attempt < max_retries - 1:
                    continue
                else:
                    return False, None, "No text could be extracted from the file"
                    
        except Exception:
            logging.exception(f"Document processing attempt {attempt + 1} failed")
            if attempt < max_retries - 1:
                continue
            else:
                return False, None, f"Error processing file after {max_retries} attempts. Please try a different file."
    
    return False, None, "Maximum retries exceeded"

# Human-readable names for the activity log (event key -> label shown in the admin page)
EVENT_LABELS = {
    'upload': 'Upload',
    'executive_summary': 'Executive summary',
    'essay_questions': 'Essay questions',
    'essay_answer': 'Essay answer marked',
    'calculation_question': 'Calculation question',
    'calculation_answer': 'Calculation answer checked',
    'multiple_choice_quiz': 'Multiple choice quiz',
    'key_concepts': 'Key concepts',
    'infographic': 'Revision infographic',
    'infographic_email': 'Infographic emailed',
    'chat': 'Chat message',
}


def record_activity(event, session_id=None, filename=None, doc_type=None, content_chars=None,
                    success=True, error=None, detail=None):
    """Append one row to the activity log. Never raises - logging must not break a feature.
    Inside a request the session id, filename and document type default to the
    current session's values."""
    try:
        if session_id is None or filename is None or doc_type is None:
            try:
                session_id = session_id or session.get('session_id')
                filename = filename or session.get('pdf_filename')
                if doc_type is None:
                    doc_type = session.get('doc_type')  # cookie only: never a DB read here
            except Exception:
                pass  # outside a request context
        fields = dict(
            event=(event or 'unknown')[:32],
            filename=(filename or '')[:256] or None,
            session_ref=(session_id or '')[:12] or None,
            doc_type=doc_type,
            content_chars=content_chars,
            success=bool(success),
            error=(str(error)[:256] if error else None),
            detail=(str(detail)[:256] if detail else None),
        )
        logging.info(f"ACTIVITY LOG: {event} | {'ok' if success else 'FAILED'} | {filename} | type={doc_type}"
                     + (f" | {detail}" if detail else ""))
        if app.testing:
            _write_activity_row(fields)          # synchronous so tests can assert immediately
        else:
            threading.Thread(target=_write_activity_row, args=(fields,), daemon=True).start()
    except Exception as e:
        logging.error(f"ACTIVITY LOG: could not record {event}: {e}")


def _write_activity_row(fields):
    """Insert one activity row (runs in a background thread so the request is not delayed)."""
    try:
        with app.app_context():
            from models import ActivityLog
            postgres_db.session.add(ActivityLog(**fields))
            postgres_db.session.commit()
    except Exception as e:
        try:
            postgres_db.session.rollback()
        except Exception:
            pass
        logging.error(f"ACTIVITY LOG: could not write {fields.get('event')}: {e}")


def record_upload(filename, session_id, success, doc_type=None, content_chars=None, error=None):
    record_activity('upload', session_id=session_id, filename=filename, doc_type=doc_type or 'unknown',
                    content_chars=content_chars, success=success, error=error)


# Signalled by the upload job as soon as the document is stored, so the upload
# route can answer "complete" directly for fast files instead of the page polling.
_UPLOAD_DONE_EVENTS = {}
_UPLOAD_DONE_LOCK = threading.Lock()
SUMMARY_STATE_KEYS = ('executive_summary', 'summary_pending')


def _upload_event(task_id, create=False):
    with _UPLOAD_DONE_LOCK:
        ev = _UPLOAD_DONE_EVENTS.get(task_id)
        if ev is None and create:
            ev = _UPLOAD_DONE_EVENTS[task_id] = threading.Event()
        return ev


def _signal_upload_done(task_id):
    ev = _upload_event(task_id)
    if ev:
        ev.set()


def _generate_summary_early(session_id, pdf_text, task_id):
    """Start the executive summary the moment the text is stored (before the
    page asks for it) and keep the result in the shared session store, where
    /summary_stream looks first."""
    started = time.perf_counter()
    storage = StorageManager()
    try:
        with app.app_context():
            tutor = TutorAI()
            tutor.set_context(pdf_text)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                text = loop.run_until_complete(tutor.generate_cheat_sheet_async())
            finally:
                loop.close()
            if text and text.strip():
                storage.store_content(session_id, 'executive_summary', text)
                store_cached_ai_result(pdf_text, 'summary', text)
                logging.info(f"Upload {task_id}: early summary ready in {time.perf_counter() - started:.1f}s")
            else:
                logging.warning(f"Upload {task_id}: early summary came back empty")
    except Exception as e:
        logging.error(f"Upload {task_id}: early summary failed (page will generate it): {e}")
    finally:
        try:
            with app.app_context():
                storage.store_content(session_id, 'summary_pending', False)
        except Exception as e:
            logging.error(f"Upload {task_id}: could not clear summary_pending: {e}")


def process_upload_background(task_id, file_data, filename, session_id):
    """Background function to process uploaded file"""
    import io
    with app.app_context():
        try:
            logging.info(f"Starting background file processing for task {task_id}")
            
            file_obj = io.BytesIO(file_data)
            file_obj.filename = filename
            
            success, pdf_text, error_message = process_document_with_fallback(file_obj)
            
            if success:
                # One write: primary_storage persists to the same table as
                # storage_manager and also fills its in-memory cache.
                primary_storage.store_content(session_id, 'pdf_content', pdf_text)

                # Reset per-document state so a fresh document starts from question 1
                # (and no document type from a previous upload lingers).
                storage_manager.store_content(session_id, 'equation_list', None)
                storage_manager.store_content(session_id, 'exam_questions', None)
                storage_manager.store_content(session_id, 'current_equation_index', 0)
                storage_manager.store_content(session_id, 'calc_doc_type', None)
                for _k in ESSAY_STATE_KEYS:
                    storage_manager.store_content(session_id, _k, None)

                # Start the executive summary NOW, in parallel with everything
                # below, so it is usually finished by the time the page asks.
                storage_manager.store_content(session_id, 'executive_summary', None)
                storage_manager.store_content(session_id, 'summary_pending', True)
                threading.Thread(target=_generate_summary_early, args=(session_id, pdf_text, task_id),
                                 daemon=True).start()

                # Report the upload complete NOW so the page can start the summary;
                # the document-type classification below runs in this thread
                # afterwards (features that need it fall back to a keyword check
                # until it is stored).
                update_task_complete(task_id, success=True, data={
                    'success': True,
                    'filename': filename,
                    'message': f'Successfully loaded: {filename}',
                    'content_length': len(pdf_text) if pdf_text else 0,
                    'document_type': None
                })
                _signal_upload_done(task_id)
                logging.info(f"Upload processing completed for task {task_id}")

                # Classify the document once (exam paper / exercise sheet / research
                # article / lecture notes) so every feature can adapt.
                doc_type = None
                try:
                    _classifier = TutorAI()
                    _classifier.set_context(pdf_text)
                    _loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(_loop)
                    try:
                        doc_type = _loop.run_until_complete(_classifier.detect_document_type_async())
                    finally:
                        _loop.close()
                    logging.info(f"Upload {task_id}: document classified as {doc_type}")
                except Exception as e:
                    logging.error(f"Upload {task_id}: document classification failed: {e}")
                storage_manager.store_content(session_id, 'calc_doc_type', doc_type)
                record_upload(filename, session_id, True, doc_type=doc_type,
                              content_chars=len(pdf_text) if pdf_text else 0)

                # Prepare calculation practice in the background so the first
                # calculation question streams immediately instead of first
                # spending ~10 s extracting the equation list / exam questions.
                try:
                    _prep = TutorAI()
                    _prep.set_context(pdf_text, doc_type=doc_type)
                    _loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(_loop)
                    try:
                        if _prep.is_question_set():
                            questions = _loop.run_until_complete(_prep.extract_exam_questions_async())
                            if questions:
                                storage_manager.store_content(session_id, 'exam_questions', questions)
                            logging.info(f"Upload {task_id}: pre-extracted {len(questions or [])} exam questions")
                        else:
                            equations = _loop.run_until_complete(_prep.extract_equation_list_async())
                            if equations:
                                storage_manager.store_content(session_id, 'equation_list', equations)
                            logging.info(f"Upload {task_id}: pre-extracted {len(equations or [])} equations")
                    finally:
                        _loop.close()
                except Exception as e:
                    logging.error(f"Upload {task_id}: calculation pre-extraction failed (will extract on demand): {e}")
            else:
                logging.error(f"Document processing failed for task {task_id}: {error_message}")
                record_upload(filename, session_id, False, error=error_message or 'Document processing failed')
                update_task_failed(task_id, error_message or 'Document processing failed')
                _signal_upload_done(task_id)
                
        except Exception as e:
            logging.error(f"Background file processing error for task {task_id}: {e}")
            record_upload(filename, session_id, False, error=str(e))
            update_task_failed(task_id, "An internal error occurred. Please try again.")
            _signal_upload_done(task_id)

@app.route('/upload', methods=['POST'])
@csrf.exempt
@rate_limit(calls_per_minute=5, use_session=True)
def upload_file():
    """Handle file upload - returns immediately with task_id"""
    try:
        init_session()

        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Enhanced filename validation with security logging
        if len(file.filename) > MAX_FILENAME_LENGTH:
            logging.warning(f"SECURITY: Filename too long from session {session.get('session_id', 'unknown')}: {len(file.filename)} characters")
            resource_monitor.increment('input_validation_failures')
            return jsonify({'error': f'Filename too long (max {MAX_FILENAME_LENGTH} characters)'}), 400

        # Use secure_filename to sanitize the filename
        original_filename = file.filename
        secure_name = secure_filename(original_filename)

        if not secure_name:
            logging.warning(f"SECURITY: Invalid filename from session {session.get('session_id', 'unknown')}: {original_filename}")
            resource_monitor.increment('input_validation_failures')
            return jsonify({'error': 'Invalid filename'}), 400

        # HTML-escape the sanitized name server-side before it appears in any response
        safe_filename = str(escape(secure_name))

        if file and secure_name and secure_name.lower().endswith(('.pdf', '.pptx')):
            
            # Reset the previous document's state here (one bulk delete) instead
            # of the page making a separate clear-session request first.
            session_id = session.get('session_id')
            clear_session_data(session_id)

            # Read file data into memory immediately
            file_data = file.read()
            
            # Generate unique task ID
            task_id = str(uuid_module.uuid4())
            
            # Initialize task status in database
            create_task(task_id, "pending")
            
            # Start background thread for file processing
            done_event = _upload_event(task_id, create=True)
            thread = threading.Thread(
                target=process_upload_background,
                args=(task_id, file_data, safe_filename, session_id),
                daemon=True
            )
            thread.start()

            # Fast path: most files are extracted and stored in well under a
            # second, so wait briefly and, if the job has already reported,
            # return the completed status in this response (no polling needed).
            completed = None
            if done_event.wait(timeout=1.5):
                try:
                    completed = get_task_status(task_id)
                except Exception:
                    completed = None
            with _UPLOAD_DONE_LOCK:
                _UPLOAD_DONE_EVENTS.pop(task_id, None)

            # Store sanitized filename in session
            session['pdf_filename'] = safe_filename
            session.pop('doc_type', None)  # new document: the type is re-classified

            # Clear previous messages (one write: DB + this worker's cache)
            try:
                primary_storage.store_content(session_id, 'messages', [])
            except Exception:
                pass  # Ignore if session doesn't exist in storage yet

            # Return task_id for polling (sanitized + escaped filename only);
            # includes the finished status when the job already completed.
            payload = {'task_id': task_id, 'filename': safe_filename}
            if completed and completed.get('status') in ('complete', 'failed'):
                payload['status'] = completed.get('status')
                payload['data'] = completed.get('data')
                payload['error'] = completed.get('error')
            return jsonify(payload)
        else:
            return jsonify({'error': 'Invalid file type. Please upload PDF or PPTX files only.'}), 400

    except Exception:
        logging.exception("Critical error in file upload")
        return jsonify({'error': 'An internal error occurred. Please try again.'}), 500

@app.route('/upload_status/<task_id>', methods=['GET'])
def get_upload_status(task_id):
    """Get the status of a file upload task"""
    try:
        # Retrieve status from database
        task_result = get_task_status(task_id)
        
        if not task_result:
            return jsonify({"status": "not_found"}), 404

        result_dict = task_result

        # If task is complete, initialize TutorAI and set context
        if result_dict.get("status") == "complete" and (result_dict.get("data") or {}).get('success'):
            session_id = session.get('session_id')
            if session_id:
                # Get PDF content
                pdf_content = (primary_storage.retrieve_content(session_id, 'pdf_content') or 
                             storage_manager.retrieve_content(session_id, 'pdf_content'))
                
                if pdf_content:
                    # Initialize TutorAI and set context
                    tutor_ai = get_tutor_ai()
                    if tutor_ai:
                        tutor_ai.set_context(pdf_content)
                        
                    # Store a flag to indicate summary generation is needed
                    session['needs_summary'] = True
                    
                    logging.info(f"Upload processing completed for task {task_id}")
                    
                    # Clean up the task from database
                    try:
                        cleanup_task(task_id)
                    except:
                        pass
        
        return jsonify(result_dict)

    except Exception:
        logging.exception("Error checking upload task status")
        return jsonify({"status": "error", "error": "An internal error occurred. Please try again."}), 500

@app.route('/start_chat_response', methods=['POST'])
@csrf.exempt
@rate_limit(calls_per_minute=30, use_session=True)
def start_chat_response():
    """Start async chat response generation using polling pattern"""
    init_session()
    
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    user_message = data['message']

    if not isinstance(user_message, str):
        resource_monitor.increment('input_validation_failures')
        return jsonify({'error': 'Message must be text'}), 400
    if len(user_message.strip()) == 0:
        resource_monitor.increment('input_validation_failures')
        return jsonify({'error': 'Message cannot be empty'}), 400
    if len(user_message) > MAX_MESSAGE_LENGTH:
        resource_monitor.increment('input_validation_failures')
        return jsonify({'error': f'Message too long (max {MAX_MESSAGE_LENGTH} characters)'}), 400
    user_message = user_message.strip()

    # Get PDF content with fallback to recent sessions
    pdf_content = get_pdf_content_with_fallback()
    if not pdf_content:
        return jsonify({'error': 'No document content found. Please upload lecture notes first.'}), 400
    
    # Get current conversation history
    session_id = session.get('session_id')
    messages = storage_manager.retrieve_content(session_id, 'messages') or []
    
    # Create conversation history in the format expected by TutorAI
    conversation_history = []
    for msg in messages:
        conversation_history.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Note: User message is already added to storage by the /chat route
    # No need to add it again here to avoid duplication
    
    try:
        # Generate unique task ID
        task_id = str(uuid_module.uuid4())
        
        # Initialize task status in PostgreSQL Database
        create_task(task_id, "pending")
        
        # Start background thread for async chat generation
        thread = threading.Thread(
            target=run_chat_response_background, 
            args=(task_id, user_message, pdf_content, conversation_history, _stored_doc_type())
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({"task_id": task_id})
        
    except Exception as e:
        logging.error(f"Error starting chat response generation: {e}")
        return jsonify({'error': 'Failed to start chat response generation'}), 500

@app.route('/chat_response_status/<task_id>')
def chat_response_status(task_id):
    """Check the status of async chat response generation"""
    try:
        # Retrieve status from PostgreSQL Database
        task_result = get_task_status(task_id)
        
        if not task_result:
            return jsonify({"status": "not_found"}), 404

        result_dict = task_result

        # If task is complete, clean up the task (no need to store in messages since frontend handles display)
        if result_dict.get("status") == "complete" and result_dict.get("success"):
            # Task completed successfully
            
            # Clean up the task from database after successful completion
            try:
                cleanup_task(task_id)
            except:
                pass
        elif result_dict.get("status") == "failed":
            logging.error(f"Chat task {task_id} failed with error: {result_dict.get('error')}")
        
        return jsonify(result_dict)

    except Exception:
        logging.exception("Error checking chat task status")
        return jsonify({"status": "error", "error": "An internal error occurred. Please try again."}), 500

@app.route('/start_essay_generation', methods=['POST'])
@csrf.exempt
@rate_limit(calls_per_minute=10, use_session=True)
def start_essay_generation():
    """Start background essay question generation using polling pattern"""
    try:
        logging.info("ESSAY POLLING: Starting essay generation")
        init_session()
        
        # Set context from stored content with fallback
        session_id = session.get('session_id')
        logging.info(f"ESSAY POLLING: Current session_id: {session_id}")
        
        pdf_content = get_pdf_content_with_fallback()
        logging.info(f"ESSAY POLLING: PDF content retrieved: {pdf_content is not None}")
        
        if not pdf_content:
            logging.error("ESSAY POLLING: No document content found even with fallback")
            return jsonify({'error': 'No document content found'}), 400

        record_activity('essay_questions')

        # Generate unique task ID
        task_id = str(uuid_module.uuid4())

        # Set initial status in PostgreSQL Database
        create_task(task_id, "pending")

        # Serve from the AI result cache when this document was already processed
        cached = get_cached_ai_result(pdf_content, 'essay')
        if cached:
            update_task_complete(task_id, success=True, data=cached)
            return jsonify({"task_id": task_id}), 202

        # Start background task in separate thread
        thread = threading.Thread(
            target=run_essay_generation_background,
            args=(task_id, pdf_content, _stored_doc_type()),
            daemon=True
        )
        thread.start()

        logging.info(f"ESSAY POLLING: Background task started with ID: {task_id}")

        return jsonify({"task_id": task_id}), 202

    except Exception:
        logging.exception("ESSAY POLLING: Critical error")
        return jsonify({"error": "Failed to start essay generation"}), 500

@app.route('/essay_status/<task_id>', methods=['GET'])
def get_essay_status(task_id):
    """Get the status of an essay generation task"""
    try:
        # Retrieve status from PostgreSQL Database
        task_result = get_task_status(task_id)
        
        if not task_result:
            return jsonify({"status": "not_found"}), 404

        result_dict = task_result

        # If task is complete, also update session storage
        if result_dict.get("status") == "complete" and result_dict.get("success"):
            session_id = session.get('session_id')
            if session_id:
                storage_manager = StorageManager()
                essay_text = result_dict.get("data", "")
                
                # Add essay to messages
                messages = storage_manager.retrieve_content(session_id, 'messages') or []
                messages.append({
                    "role": "assistant", 
                    "content": f"📝 **Essay Question Practice:**\n\n{essay_text}"
                })
                storage_manager.store_content(session_id, 'messages', messages)
                
                logging.info(f"ESSAY POLLING: Essay stored in session for task {task_id}")
                
                # Clean up the task from database after successful completion
                try:
                    cleanup_task(task_id)
                except:
                    pass
        
        return jsonify(result_dict)

    except Exception:
        logging.exception(f"ESSAY POLLING: Error getting task status for {task_id}")
        return jsonify({"error": "Failed to get task status"}), 500

@app.route('/start_key_concepts_generation', methods=['POST'])
@csrf.exempt
@rate_limit(calls_per_minute=10, use_session=True)
def start_key_concepts_generation():
    """Start background key concepts explanation generation using polling pattern"""
    try:
        logging.info("KEY CONCEPTS POLLING: Starting key concepts generation")
        init_session()
        
        # Set context from stored content with fallback
        session_id = session.get('session_id')
        logging.info(f"KEY CONCEPTS POLLING: Current session_id: {session_id}")
        
        pdf_content = get_pdf_content_with_fallback()
        logging.info(f"KEY CONCEPTS POLLING: PDF content retrieved: {pdf_content is not None}")
        
        if not pdf_content:
            logging.error("KEY CONCEPTS POLLING: No document content found even with fallback")
            return jsonify({'error': 'No document content found'}), 400

        record_activity('key_concepts')

        # Generate unique task ID
        task_id = str(uuid_module.uuid4())

        # Set initial status in PostgreSQL Database
        create_task(task_id, "pending")

        # Serve from the AI result cache when this document was already processed
        cached = get_cached_ai_result(pdf_content, 'key_concepts')
        if cached:
            update_task_complete(task_id, success=True, data=cached)
            return jsonify({"task_id": task_id}), 202

        # Start background task in separate thread
        thread = threading.Thread(
            target=run_key_concepts_generation_background,
            args=(task_id, pdf_content, _stored_doc_type()),
            daemon=True
        )
        thread.start()

        logging.info(f"KEY CONCEPTS POLLING: Background task started with ID: {task_id}")

        return jsonify({"task_id": task_id}), 202

    except Exception:
        logging.exception("KEY CONCEPTS POLLING: Critical error")
        return jsonify({"error": "Failed to start key concepts generation"}), 500

@app.route('/key_concepts_status/<task_id>', methods=['GET'])
def get_key_concepts_status(task_id):
    """Get the status of a key concepts explanation generation task"""
    try:
        # Retrieve status from PostgreSQL Database
        task_result = get_task_status(task_id)
        
        if not task_result:
            return jsonify({"status": "not_found"}), 404

        result_dict = task_result

        # If task is complete, also update session storage
        if result_dict.get("status") == "complete" and result_dict.get("success"):
            session_id = session.get('session_id')
            if session_id:
                storage_manager = StorageManager()
                key_concepts_text = result_dict.get("data", "")
                
                # Add key concepts explanation to messages
                messages = storage_manager.retrieve_content(session_id, 'messages') or []
                messages.append({
                    "role": "assistant", 
                    "content": f"🔑 **Key Concepts Explained:**\n\n{key_concepts_text}"
                })
                storage_manager.store_content(session_id, 'messages', messages)
                
                logging.info(f"KEY CONCEPTS POLLING: Key concepts stored in session for task {task_id}")
                
                # Clean up the task from database after successful completion
                try:
                    cleanup_task(task_id)
                except:
                    pass
        
        return jsonify(result_dict)

    except Exception:
        logging.exception(f"KEY CONCEPTS POLLING: Error getting task status for {task_id}")
        return jsonify({"error": "Failed to get task status"}), 500

@app.route('/start_infographic_generation', methods=['POST'])
@csrf.exempt
@rate_limit(calls_per_minute=5, use_session=True)
def start_infographic_generation():
    """Start background infographic image generation using polling pattern"""
    try:
        logging.info("INFOGRAPHIC POLLING: Starting infographic generation")
        init_session()

        pdf_content = get_pdf_content_with_fallback()
        logging.info(f"INFOGRAPHIC POLLING: PDF content retrieved: {pdf_content is not None}")

        if not pdf_content:
            logging.error("INFOGRAPHIC POLLING: No document content found even with fallback")
            return jsonify({'error': 'No document content found'}), 400

        record_activity('infographic')

        # Generate unique task ID
        task_id = str(uuid_module.uuid4())

        # Set initial status in PostgreSQL Database
        create_task(task_id, "pending")

        # Serve from the AI result cache when this document was already processed
        cached = get_cached_ai_result(pdf_content, 'infographic')
        if cached:
            update_task_complete(task_id, success=True, data=cached)
            return jsonify({"task_id": task_id}), 202

        # Start background task in separate thread
        thread = threading.Thread(
            target=run_infographic_generation_background,
            args=(task_id, pdf_content, _stored_doc_type()),
            daemon=True
        )
        thread.start()

        logging.info(f"INFOGRAPHIC POLLING: Background task started with ID: {task_id}")

        return jsonify({"task_id": task_id}), 202

    except Exception:
        logging.exception("INFOGRAPHIC POLLING: Critical error")
        return jsonify({"error": "Failed to start infographic generation"}), 500

@app.route('/email_infographic', methods=['POST'])
@csrf.exempt
@rate_limit(calls_per_minute=5, use_session=True)
def email_infographic_route():
    """Email the revision infographic to the user as a PDF.

    Body: {"email": "...", "task_id": "<optional>", "image_b64": "<optional fallback>"}
    - If the task is still running, the address is registered and the PDF is sent
      by the background thread when the image is complete ("scheduled").
    - Otherwise the image is taken from the completed task, the AI result cache,
      or the client-supplied fallback, and sent immediately ("sent").
    """
    if not is_email_configured():
        return jsonify({'error': 'Email delivery is not set up on this server yet.'}), 503
    try:
        payload = request.get_json(silent=True) or {}
        email = normalise_email(payload.get('email'))
        if not email:
            return jsonify({'error': 'Please enter a valid email address.'}), 400
        task_id = payload.get('task_id')
        image_b64 = payload.get('image_b64')
        document_name = session.get('pdf_filename')
        session['infographic_email'] = email
        record_activity('infographic_email', detail='requested')
        transport = email_transport()
        logging.info(f"INFOGRAPHIC EMAIL: request received (transport={transport}, task={task_id}, "
                     f"client_image={'yes' if image_b64 else 'no'})")

        if task_id:
            task = get_task_status(task_id)
            if task and task.get('status') == 'pending':
                register_infographic_email(task_id, email, document_name)
                # The task may have completed while we were registering (the
                # background thread checks for a request only once, right after
                # finishing). Re-check; if it is done now, send immediately.
                task = get_task_status(task_id)
                if not (task and task.get('status') == 'complete'):
                    logging.info(f"INFOGRAPHIC EMAIL: scheduled delivery for task {task_id}")
                    return jsonify({'status': 'scheduled', 'email': email, 'transport': transport})
                if pop_infographic_email(task_id) is None:
                    # Background thread already took the request and is sending it
                    return jsonify({'status': 'scheduled', 'email': email, 'transport': transport})
            if task and task.get('status') == 'complete' and task.get('success') and task.get('data'):
                image_b64 = task['data']

        if not image_b64:
            pdf_content = get_pdf_content_with_fallback()
            image_b64 = get_cached_ai_result(pdf_content, 'infographic') if pdf_content else None

        if not image_b64 or not isinstance(image_b64, str):
            return jsonify({'error': 'No infographic is available to send. Please generate it first.'}), 404

        try:
            email_infographic(email, image_b64, document_name=document_name)
        except Exception as e:
            logging.error(f"INFOGRAPHIC EMAIL: send failed: {e}")
            return jsonify({'error': _email_error_message(e)}), 502

        logging.info(f"INFOGRAPHIC EMAIL: sent immediately via {transport}")
        return jsonify({'status': 'sent', 'email': email, 'transport': transport})
    except Exception:
        logging.exception("INFOGRAPHIC EMAIL: unexpected error")
        return jsonify({'error': 'Could not email the infographic.'}), 500

@app.route('/infographic_email_status/<task_id>', methods=['GET'])
def infographic_email_status(task_id):
    """Outcome of a scheduled 'email me when ready' delivery: pending / sent / failed."""
    try:
        with app.app_context():
            result = StorageManager().retrieve_content(task_id, _INFOGRAPHIC_EMAIL_RESULT_TYPE)
        if isinstance(result, dict) and result.get('status'):
            return jsonify(result)
        return jsonify({'status': 'pending'})
    except Exception:
        logging.exception("INFOGRAPHIC EMAIL: status lookup failed")
        return jsonify({'status': 'pending'})

@app.route('/infographic_status/<task_id>', methods=['GET'])
def get_infographic_status(task_id):
    """Get the status of an infographic generation task"""
    try:
        # Retrieve status from PostgreSQL Database
        task_result = get_task_status(task_id)

        if not task_result:
            return jsonify({"status": "not_found"}), 404

        # The payload is a large base64 image, so unlike the text features it
        # is NOT appended to the session chat history; just clean the task up
        # once the result has been delivered.
        if task_result.get("status") == "complete" and task_result.get("success"):
            try:
                cleanup_task(task_id)
            except:
                pass

        return jsonify(task_result)

    except Exception:
        logging.exception(f"INFOGRAPHIC POLLING: Error getting task status for {task_id}")
        return jsonify({"error": "Failed to get task status"}), 500

def clear_session_data(session_id=None):
    """Clear current session data from both Flask session and file storage"""
    if session_id is None:
        session_id = session.get('session_id')
    
    if session_id:
        # Clear all session-related files from storage
        storage_manager = StorageManager()
        content_types = [
            'pdf_content', 'quiz_data', 'calculation_quiz_data', 
            'current_calculation_question', 'used_calculation_questions',
            'messages', 'quiz_questions', 'practice_equations',
            'equation_list', 'exam_questions', 'calc_doc_type', 'current_equation_index',
            'current_essay_question', 'essay_mode_active', 'used_essay_questions',
            'infographic_email_request', 'infographic_email_result',
            'executive_summary', 'summary_pending'
        ]
        
        # One DELETE for all types (this used to be 17 sequential round trips,
        # which made the pre-upload clear take over two seconds).
        try:
            storage_manager.delete_many(session_id, content_types)
        except Exception as e:
            logging.debug(f"Could not clear session content for {session_id}: {e}")
    
    # Clear the Flask session but KEEP the session id. Regenerating it here made
    # any request that raced the clear (e.g. an upload started in the same
    # moment) store content under the old id while later requests looked it up
    # under a new one -> "Document content not available" right after a
    # successful upload.
    session.clear()
    if session_id:
        session['session_id'] = session_id

@app.route('/clear_session', methods=['POST'])
@csrf.exempt
def clear_session():
    """Clear current session"""
    clear_session_data()
    return jsonify({'success': True})

def _admin_authorised():
    """True when the request carries valid admin credentials.

    Browser access: HTTP Basic authentication (the browser shows a username /
    password prompt) checked against the ADMIN_USER (default "admin") and
    ADMIN_PASSWORD secrets. Scripts may instead send an X-Admin-Token header
    equal to the ADMIN_TOKEN secret. Nothing is accepted in the URL.
    """
    password = os.environ.get('ADMIN_PASSWORD', '')
    user = os.environ.get('ADMIN_USER', 'admin')
    auth = request.authorization
    if password and auth and auth.type == 'basic' and auth.username is not None and auth.password is not None:
        if hmac.compare_digest(auth.username, user) and hmac.compare_digest(auth.password, password):
            return True
    token = os.environ.get('ADMIN_TOKEN', '').strip()
    supplied = (request.headers.get('X-Admin-Token') or '').strip()
    if token and supplied and hmac.compare_digest(token, supplied):
        return True
    return False


def _admin_challenge():
    """401 with a Basic-auth challenge so the browser prompts for the password.
    If no admin password is configured at all, behave as if the page did not exist."""
    if not os.environ.get('ADMIN_PASSWORD', '') and not os.environ.get('ADMIN_TOKEN', '').strip():
        return jsonify({'error': 'Not found'}), 404
    return Response('Authentication required', 401,
                    {'WWW-Authenticate': 'Basic realm="QUB Finance AI Tutor admin", charset="UTF-8"'})


@app.route('/admin/activity')
@app.route('/admin/uploads')
@csrf.exempt
def admin_activity():
    """Activity log for the module team: uploads and every feature use, newest first.
    Password protected (HTTP Basic auth, ADMIN_USER / ADMIN_PASSWORD secrets).
    ?event=upload (or any event key) filters; ?format=csv downloads; ?limit=N (default 500)."""
    if not _admin_authorised():
        return _admin_challenge()
    try:
        from models import ActivityLog
        from datetime import timezone as _tz
        limit = max(1, min(int(request.args.get('limit', 500)), 5000))
        event_filter = (request.args.get('event') or '').strip()
        q = ActivityLog.query
        if event_filter:
            q = q.filter(ActivityLog.event == event_filter)
        rows = q.order_by(ActivityLog.occurred_at.desc()).limit(limit).all()
        try:
            from zoneinfo import ZoneInfo
            display_tz, tz_label = ZoneInfo('Europe/London'), 'Europe/London'
        except Exception:  # no tz database (e.g. Windows without tzdata): show UTC
            display_tz, tz_label = _tz.utc, 'UTC'

        def local_time(dt):
            return dt.replace(tzinfo=_tz.utc).astimezone(display_tz).strftime('%Y-%m-%d %H:%M:%S') if dt else ''

        def label(ev):
            return EVENT_LABELS.get(ev, ev)

        if request.args.get('format') == 'csv':
            import csv
            import io as _io
            buf = _io.StringIO()
            w = csv.writer(buf)
            w.writerow([f'time_{tz_label}', 'event', 'filename', 'document_type', 'status', 'error', 'detail',
                        'text_chars', 'session_ref'])
            for r in rows:
                w.writerow([local_time(r.occurred_at), label(r.event), r.filename or '', r.doc_type or '',
                            'ok' if r.success else 'failed', r.error or '', r.detail or '',
                            r.content_chars or '', r.session_ref or ''])
            return Response(buf.getvalue(), mimetype='text/csv',
                            headers={'Content-Disposition': 'attachment; filename="activity_log.csv"'})

        failed_cell = '<span style="color:#b00">failed</span>'
        cells = ''.join(
            f"<tr><td>{escape(local_time(r.occurred_at))}</td><td>{escape(label(r.event))}</td>"
            f"<td>{escape(r.filename or '')}</td><td>{escape(r.doc_type or '')}</td>"
            f"<td>{'ok' if r.success else failed_cell}</td>"
            f"<td>{escape(r.error or r.detail or '')}</td><td>{escape(r.session_ref or '')}</td></tr>"
            for r in rows
        )
        filters = ' '.join(
            f'<a href="?event={k}">{escape(v)}</a>' for k, v in EVENT_LABELS.items()
        )
        html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Activity log</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;color:#222}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}}th{{background:#f4f4f4}}tr:nth-child(even){{background:#fafafa}}
.f a{{margin-right:.8rem}}</style></head>
<body><h2>Activity log</h2>
<p>{len(rows)} most recent event(s){' - ' + escape(label(event_filter)) if event_filter else ''}, times in {tz_label}.
<a href="?format=csv{'&amp;event=' + escape(event_filter) if event_filter else ''}">Download CSV</a></p>
<p class="f">Filter: <a href="?">All</a> {filters}</p>
<table><tr><th>Time</th><th>Event</th><th>Document</th><th>Type</th><th>Status</th><th>Note</th><th>Session</th></tr>{cells}</table>
</body></html>"""
        return Response(html, mimetype='text/html')
    except Exception:
        logging.exception("ACTIVITY LOG: could not render admin view")
        return jsonify({'error': 'Could not load the activity log'}), 500

@app.route('/security_metrics')
@csrf.exempt
def security_metrics():
    """Security monitoring endpoint - localhost access only"""
    if request.remote_addr not in ('127.0.0.1', '::1'):
        return jsonify({'error': 'Access denied'}), 403
    try:
        metrics = resource_monitor.get_metrics()
        security_data = {
            'security_violations': metrics.get('security_violations', 0),
            'input_validation_failures': metrics.get('input_validation_failures', 0),
            'csrf_failures': metrics.get('csrf_failures', 0),
            'total_requests': metrics.get('requests_processed', 0),
            'error_rate': metrics.get('errors', 0),
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(security_data)
    except Exception as e:
        logging.error(f"Error getting security metrics: {e}")
        return jsonify({'error': 'Unable to fetch security metrics'}), 500

# Initialize database tables
with app.app_context():
    # Import models to register them with SQLAlchemy
    import models  # noqa: F401
    
    # Create all tables
    postgres_db.create_all()
    logging.info("PostgreSQL tables created successfully")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
