from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, Response
from flask_compress import Compress
from werkzeug.middleware.proxy_fix import ProxyFix
from database import postgres_db
from werkzeug.utils import secure_filename
from flask_wtf.csrf import CSRFProtect, validate_csrf
from flask_wtf import FlaskForm
import os
import json
import logging
import threading
import uuid as uuid_module
from pdf_processor import extract_text_from_file
from tutor_ai import TutorAI
from database_storage_manager import DatabaseStorageManager as StorageManager
from performance_optimizations import optimized_storage, resource_monitor, rate_limit, start_periodic_cleanup
from speed_optimizations import fast_ai_client, preload_critical_resources, optimize_json_responses, speed_metrics
import tempfile
# ReplitDB no longer needed - using PostgreSQL for task tracking

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")
if not app.secret_key:
    raise RuntimeError("SESSION_SECRET environment variable must be set")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)  # needed for url_for to generate with https

# Configure PostgreSQL database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
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
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour session timeout

# Configure session
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Security configuration
MAX_MESSAGE_LENGTH = 5000  # Maximum characters for chat messages
MAX_FILENAME_LENGTH = 255  # Maximum filename length

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize storage managers - use optimized version for better performance
storage_manager = StorageManager()  # Keep for backward compatibility
primary_storage = optimized_storage  # Use optimized version as primary

# Start performance monitoring and cleanup
start_periodic_cleanup()

# Configure for production deployment
from deployment_config import configure_for_production
import os
from datetime import datetime
from flask import request

if os.environ.get('FLASK_ENV') == 'production':
    app = configure_for_production(app)

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
    """Get or create TutorAI instance"""
    if session.get('tutor_ai') is None:
        try:
            tutor_ai = TutorAI()
            session['tutor_ai'] = True  # Just mark as initialized
            return tutor_ai
        except Exception as e:
            logging.error(f"Failed to initialize TutorAI: {e}")
            return None
    return TutorAI()

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
    Background function to generate calculation questions using async gpt-5.4-mini.
    On first call, extracts an ordered equation list from the notes and stores it in the
    session. Subsequent calls use the stored list and advance the index.
    """
    with app.app_context():
        try:
            logging.info(f"BACKGROUND TASK: Starting calculation generation for task {task_id}")

            tutor_ai = TutorAI()
            tutor_ai.set_context(pdf_content)

            # Access storage inside app context (before entering async)
            _storage = StorageManager()
            equation_list = _storage.retrieve_content(session_id, 'equation_list')
            current_index = _storage.retrieve_content(session_id, 'current_equation_index') or 0

            import asyncio

            async def async_generation():
                nonlocal equation_list, current_index

                # --- Equation list: extract on first use ---
                if not equation_list:
                    logging.info("BACKGROUND TASK: No equation list found — extracting from notes")
                    equation_list = await tutor_ai.extract_equation_list_async()
                    logging.info(f"BACKGROUND TASK: Extraction returned {len(equation_list) if equation_list else 0} equations")
                    if not equation_list:
                        return "I couldn't find any calculable equations in your notes. Please make sure your document contains mathematical formulas."
                    _storage.store_content(session_id, 'equation_list', equation_list)
                    _storage.store_content(session_id, 'current_equation_index', 0)
                    current_index = 0
                    logging.info(f"BACKGROUND TASK: Stored {len(equation_list)} equations")

                # --- Pick the current equation ---
                if current_index >= len(equation_list):
                    return f"You've worked through all {len(equation_list)} equations in your notes — great work! You can upload new notes to continue practising."

                specific_equation = equation_list[current_index]
                logging.info(f"BACKGROUND TASK: Generating question for equation {current_index + 1}/{len(equation_list)}: {specific_equation[:80]}")

                return await tutor_ai.generate_calculation_question_async(specific_equation=specific_equation)

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
            update_task_failed(task_id, str(e))

def run_calculation_answer_check_background(task_id, challenge_question, user_answer, pdf_content):
    """
    Background function to check calculation answers using async gpt-5.4-mini.
    This runs in a separate thread to avoid blocking the web server.
    """
    try:
        logging.info(f"CALC ANSWER BACKGROUND: Starting answer check for task {task_id}")
        
        # Initialize TutorAI and set context
        tutor_ai = TutorAI()
        tutor_ai.set_context(pdf_content)
        
        # Check calculation answer using async method with gpt-5.4-mini
        import asyncio
        
        async def async_answer_check():
            return await tutor_ai.check_calculation_answer_async(challenge_question, user_answer)
        
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
        update_task_failed(task_id, str(e))

def run_summary_generation_background(task_id, pdf_content):
    """
    Background function to generate executive summaries using async Gemini.
    This runs in a separate thread to avoid blocking the web server.
    """
    try:
        logging.info(f"SUMMARY BACKGROUND: Starting summary generation for task {task_id}")
        
        # Initialize TutorAI and set context
        tutor_ai = TutorAI()
        tutor_ai.set_context(pdf_content)
        
        # Generate summary using async method with Gemini
        import asyncio
        
        async def async_summary_generation():
            return await tutor_ai.generate_cheat_sheet_async()
        
        # Run the async function in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_summary_generation())
        loop.close()
        
        logging.info(f"SUMMARY BACKGROUND: Generation completed for task {task_id}")
        
        # Store the final result in PostgreSQL Database
        update_task_complete(task_id, success=True, data=result)
        
    except Exception as e:
        logging.error(f"SUMMARY BACKGROUND: Error in task {task_id}: {e}")
        update_task_failed(task_id, str(e))

def run_essay_generation_background(task_id, pdf_content):
    """
    Background function to generate essay questions using async Gemini.
    This runs in a separate thread to avoid blocking the web server.
    """
    try:
        logging.info(f"ESSAY BACKGROUND: Starting essay generation for task {task_id}")
        
        # Initialize TutorAI and set context
        tutor_ai = TutorAI()
        tutor_ai.set_context(pdf_content)
        
        # Generate essay using async method with Gemini
        import asyncio
        
        async def async_essay_generation():
            return await tutor_ai.generate_essay_question_async()
        
        # Run the async function in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_essay_generation())
        loop.close()
        
        logging.info(f"ESSAY BACKGROUND: Generation completed for task {task_id}")
        
        # Store the final result in PostgreSQL Database
        update_task_complete(task_id, success=True, data=result)
        
    except Exception as e:
        logging.error(f"ESSAY BACKGROUND: Error in task {task_id}: {e}")
        update_task_failed(task_id, str(e))

def run_key_concepts_generation_background(task_id, pdf_content):
    """
    Background function to generate key concepts explanations using async Gemini.
    This runs in a separate thread to avoid blocking the web server.
    """
    try:
        logging.info(f"KEY CONCEPTS BACKGROUND: Starting key concepts generation for task {task_id}")
        
        # Initialize TutorAI and set context
        tutor_ai = TutorAI()
        tutor_ai.set_context(pdf_content)
        
        # Generate key concepts using async method with Gemini
        import asyncio
        
        async def async_key_concepts_generation():
            return await tutor_ai.explain_key_concepts_async()
        
        # Run the async function in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_key_concepts_generation())
        loop.close()
        
        logging.info(f"KEY CONCEPTS BACKGROUND: Generation completed for task {task_id}")
        
        # Store the final result in PostgreSQL Database
        update_task_complete(task_id, success=True, data=result)
        
    except Exception as e:
        logging.error(f"KEY CONCEPTS BACKGROUND: Error in task {task_id}: {e}")
        update_task_failed(task_id, str(e))

def run_chat_response_background(task_id, user_message, pdf_content, conversation_history):
    """
    Background function to generate chat responses using async processing.
    This runs in a separate thread to avoid blocking the web server.
    """
    try:
        logging.info(f"CHAT BACKGROUND: Starting chat response generation for task {task_id}")
        
        # Initialize TutorAI and set context
        tutor_ai = TutorAI()
        tutor_ai.set_context(pdf_content)
        tutor_ai.conversation_history = conversation_history or []
        
        # Generate chat response using async method
        import asyncio
        
        async def async_chat_generation():
            return await tutor_ai.get_response_async(user_message)
        
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
        update_task_failed(task_id, str(e))

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
                         message_count=len(messages),
                         quiz_active=session.get('quiz_active', False),
                         equation_active=session.get('equation_active', False))

@app.route('/load_messages')
def load_messages():
    """DISABLED: This endpoint was causing content reloading issues"""
    logging.info("LOAD_MESSAGES: Endpoint disabled to prevent content reloading")
    return jsonify({'success': False, 'messages': []}), 200

@app.route('/simple_chat', methods=['POST'])
@csrf.exempt
@rate_limit(calls_per_minute=30, use_session=True)  # Session-based rate limiting for chat
def simple_chat():
    """Simple async chat endpoint that calls Gemini asynchronously using asyncio"""
    import asyncio
    
    async def async_chat_handler():
        """Async handler for chat processing"""
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
                    # Detect numerical answers (numbers, percentages, currency)
                    import re
                    answer_pattern = r'^[£$€¥]?[\d,]+\.?\d*%?$|^\d+\.?\d*%?$|^[\d,]+\.?\d*[£$€¥]?$'
                    if re.match(answer_pattern, user_message.strip()) or any(char.isdigit() for char in user_message):
                        # This looks like a numerical answer - use the existing calculation answer checking
                        logging.info(f"SIMPLE_CHAT: Detected numerical answer in calculation mode: {user_message}")
                        # We'll continue with normal processing but use calculation checking logic
                    else:
                        # Not a numerical answer, provide guidance
                        return {
                            'success': True,
                            'response': 'Please provide a numerical answer to the calculation question above, or type "skip" to get a new question, or "end practice" to finish. 🔢'
                        }, 200
            
            # Get document context using fallback mechanism
            pdf_content = get_pdf_content_with_fallback()
            
            if not pdf_content:
                logging.info("No document content found for simple chat")
                return {
                    'success': False,
                    'error': 'I need you to upload your lecture notes first before I can help you study! 📚'
                }, 400
            
            # Create tutor AI instance and set context
            tutor_ai = TutorAI()
            tutor_ai.set_context(pdf_content)
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
                # Normal chat mode - get response asynchronously using gpt-5.4-mini primary with Gemini fallback
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
                
        except Exception as e:
            logging.error(f"SIMPLE_CHAT: Error processing chat: {e}")
            return {
                'success': False,
                'error': str(e)
            }, 500
    
    # Run async function in event loop
    try:
        result, status_code = asyncio.run(async_chat_handler())
        return jsonify(result), status_code
    except Exception as e:
        logging.error(f"SIMPLE_CHAT: Error in asyncio.run: {e}")
        return jsonify({
            'success': False,
            'error': f'Async processing error: {str(e)}'
        }), 500

@app.route('/simple_chat_stream', methods=['POST'])
@csrf.exempt
@rate_limit(calls_per_minute=30, use_session=True)
def simple_chat_stream():
    """Streaming chat endpoint using Server-Sent Events."""
    import asyncio
    import queue

    init_session()

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

    # Check calculation mode — fall back to non-streaming simple_chat for calc answers
    calculation_mode = storage_manager.retrieve_content(session_id, 'calculation_mode_active')
    current_calculation = storage_manager.retrieve_content(session_id, 'current_calculation_question')
    if calculation_mode and current_calculation:
        # Delegate to the non-streaming endpoint for calculation answers
        return simple_chat()

    pdf_content = get_pdf_content_with_fallback()
    if not pdf_content:
        return jsonify({'success': False, 'error': 'I need you to upload your lecture notes first before I can help you study! 📚'}), 400

    tutor_ai = TutorAI()
    tutor_ai.set_context(pdf_content)

    stored_messages = storage_manager.retrieve_content(session_id, 'messages') or []
    for msg in stored_messages:
        tutor_ai.conversation_history.append({'role': msg['role'], 'content': msg['content']})

    # Use a thread to run the async generator and push chunks into a queue
    q = queue.Queue()

    def _run_stream():
        async def _consume():
            full_response = ""
            try:
                async for chunk in tutor_ai.get_response_stream_async(user_message):
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
    import asyncio
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

    q = queue_mod.Queue()

    def _run_stream():
        async def _consume():
            full_response = ""
            try:
                tutor_ai = TutorAI()
                tutor_ai.set_context(pdf_content)

                if action == 'key_concepts':
                    gen = tutor_ai.explain_key_concepts_stream_async()
                else:
                    gen = tutor_ai.generate_essay_question_stream_async()

                async for chunk in gen:
                    full_response += chunk
                    q.put(chunk)
            except Exception as e:
                logging.error(f"QUICKACTION STREAM: Error during streaming ({action}): {e}")
                if not full_response:
                    q.put(f"I'm having trouble right now. Please try again in a moment.")
            finally:
                # Store the response in message history
                try:
                    storage_manager = StorageManager()
                    msgs = storage_manager.retrieve_content(session_id, 'messages') or []
                    msgs.append({'role': 'user', 'content': 'Explanation of key concepts' if action == 'key_concepts' else 'Essay question'})
                    msgs.append({'role': 'assistant', 'content': full_response})
                    storage_manager.store_content(session_id, 'messages', msgs)
                except Exception as e:
                    logging.error(f"QUICKACTION STREAM: Error storing messages: {e}")
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

def generate_summary_with_fallback(tutor_ai, pdf_content, max_retries=3):
    """
    Generate executive summary with fallback mechanism
    Returns: (success, summary, error_message)
    """
    for attempt in range(max_retries):
        try:
            # Set context and generate summary
            tutor_ai.set_context(pdf_content)
            summary = tutor_ai.generate_cheat_sheet()
            
            if summary and summary.strip():
                return True, summary, None
            else:
                if attempt < max_retries - 1:
                    continue
                else:
                    return False, None, "Generated summary was empty"
                    
        except Exception as e:
            logging.error(f"Summary generation attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                return False, None, f"Error generating summary after {max_retries} attempts: {str(e)}"
    
    return False, None, "Maximum retries exceeded"

@app.route('/start_summary_generation', methods=['POST'])
@csrf.exempt
def start_summary_generation():
    """Start background summary generation using polling pattern"""
    # Import db at function level to avoid scoping issues
    # ReplitDB no longer needed - using PostgreSQL for task tracking
    
    try:
        init_session()
        
        # Set context from stored content with fallback
        session_id = session.get('session_id')
        
        # Add delay to handle race condition with file upload
        import time
        time.sleep(2.0)  # 2 second delay to allow file storage to complete and database consistency
        
        pdf_content = get_pdf_content_with_fallback()
        
        if not pdf_content:
            logging.error("No document content found for summary generation")
            
            # Check if content exists with direct database query using storage manager
            try:
                storage_manager = StorageManager()
                pdf_content = storage_manager.retrieve_content(session_id, 'pdf_content')
            except Exception as storage_error:
                logging.error(f"Storage manager query error: {storage_error}")
            
            if not pdf_content:
                # Since summary generation now only starts AFTER upload completion confirmation,
                # we should always have content available. One more retry with longer delay.
                time.sleep(5.0)
                pdf_content = get_pdf_content_with_fallback()
                
                if not pdf_content:
                    # Try one more time with storage manager directly
                    try:
                        storage_manager = StorageManager()
                        pdf_content = storage_manager.retrieve_content(session_id, 'pdf_content')
                    except:
                        pass
                    
                    if not pdf_content:
                        logging.error("No content found even after upload confirmation")
                        # Return 200 with error message instead of 400 to prevent HTTP errors in frontend
                        return jsonify({'success': False, 'error': 'Document content not available. Please try uploading your file again.'}), 200
        
        # Generate unique task ID
        task_id = str(uuid_module.uuid4())
        
        # Set initial status in PostgreSQL Database
        create_task(task_id, "pending")
        
        # Start background task in separate thread
        thread = threading.Thread(
            target=run_summary_generation_background, 
            args=(task_id, pdf_content)
        )
        thread.start()
        
        
        return jsonify({"task_id": task_id}), 202
        
    except Exception as e:
        logging.error(f"Critical error in summary generation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/summary_status/<task_id>', methods=['GET'])
def get_summary_status(task_id):
    """Get the status of a summary generation task"""
    try:
        # Retrieve status from PostgreSQL Database
        task_result = get_task_status(task_id)
        
        if not task_result:
            return jsonify({"status": "not_found"}), 404
        
        # Convert ObservedDict to regular dict for JSON serialization
        result_dict = dict(task_result)
        
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
        
    except Exception as e:
        logging.error(f"SUMMARY POLLING: Error checking task status: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

# Keep old route for backward compatibility, but redirect to new async approach
@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    """Legacy route - redirects to new async approach"""
    return start_summary_generation()



@app.route('/explain_concepts', methods=['POST'])
def explain_concepts():
    """Legacy route - redirects to new async approach"""
    return start_key_concepts_generation()

def run_quiz_generation_background(task_id, pdf_content):
    """Background task to generate retrieval quiz using async methods"""
    import asyncio
    import threading
    
    def run_async():
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async function
            result = loop.run_until_complete(run_quiz_generation_async(task_id, pdf_content))
            return result
        finally:
            loop.close()
    
    return run_async()

async def run_quiz_generation_async(task_id, pdf_content):
    """Async worker function for quiz generation"""
    try:
        logging.info(f"ASYNC QUIZ WORKER: Starting async quiz generation for task {task_id}")
        
        # Update status to running (we'll handle this by updating with partial status)
        # Note: PostgreSQL model handles running state differently
        pass  # Running state will be implicit between pending and complete
        
        # Create TutorAI instance and set context
        tutor_ai = TutorAI()
        tutor_ai.set_context(pdf_content)
        
        # Generate quiz using async method
        quiz_questions = await tutor_ai.generate_retrieval_quiz_async()
        
        if quiz_questions:
            logging.info(f"ASYNC QUIZ WORKER: Successfully generated {len(quiz_questions)} questions for task {task_id}")
            # Convert to regular Python objects if they're ObservedList/ObservedDict to avoid JSON serialization errors
            def convert_observed_objects(obj):
                """Recursively convert ObservedList/ObservedDict to regular Python objects"""
                if hasattr(obj, '__iter__') and not isinstance(obj, str):
                    if hasattr(obj, 'items'):  # dict-like
                        return {k: convert_observed_objects(v) for k, v in obj.items()}
                    else:  # list-like
                        return [convert_observed_objects(item) for item in obj]
                return obj
            
            quiz_questions = convert_observed_objects(quiz_questions)
            # Store successful result
            update_task_complete(task_id, success=True, data=quiz_questions)
        else:
            logging.error(f"ASYNC QUIZ WORKER: No questions generated for task {task_id}")
            update_task_failed(task_id, "No quiz questions could be generated. Please try again.")
            
    except Exception as e:
        logging.error(f"ASYNC QUIZ WORKER: Error in quiz generation for task {task_id}: {e}")
        update_task_failed(task_id, f"Quiz generation failed: {str(e)}")

@app.route('/start_quiz_generation', methods=['POST'])
@csrf.exempt
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
        
        # Generate unique task ID
        task_id = str(uuid_module.uuid4())
        
        # Set initial status in PostgreSQL Database
        create_task(task_id, "pending")
        
        # Start background task in separate thread
        thread = threading.Thread(
            target=run_quiz_generation_background, 
            args=(task_id, pdf_content)
        )
        thread.start()
        
        logging.info(f"QUIZ POLLING: Background task started with ID: {task_id}")
        
        return jsonify({"task_id": task_id}), 202
        
    except Exception as e:
        logging.error(f"QUIZ POLLING: Critical error: {e}")
        import traceback
        logging.error(f"QUIZ POLLING: Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/quiz_status/<task_id>', methods=['GET'])
def get_quiz_status(task_id):
    """Get the status of a quiz generation task"""
    try:
        # Retrieve status from PostgreSQL Database
        task_result = get_task_status(task_id)
        
        if not task_result:
            return jsonify({"status": "not_found"}), 404
        
        # Convert ObservedDict/ObservedList to regular Python objects for JSON serialization
        def convert_observed_objects(obj):
            """Recursively convert ObservedList/ObservedDict to regular Python objects"""
            if hasattr(obj, '__iter__') and not isinstance(obj, str):
                if hasattr(obj, 'items'):  # dict-like
                    return {k: convert_observed_objects(v) for k, v in obj.items()}
                else:  # list-like
                    return [convert_observed_objects(item) for item in obj]
            return obj
        
        result_dict = convert_observed_objects(task_result)
        
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
        
    except Exception as e:
        logging.error(f"Error checking quiz task status: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

# Keep old route for backward compatibility, but redirect to new async approach
@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    """Legacy route - redirects to new async approach"""
    return start_quiz_generation()

@app.route('/generate_essay', methods=['POST'])
def generate_essay():
    """Legacy route - redirects to new async approach"""
    return start_essay_generation()

# Removed list_equations route - now using simplified direct calculation approach

# Removed old calculation quiz routes - now using simplified chat-based approach

@app.route('/start_calculation_generation', methods=['POST'])
@csrf.exempt
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
        
        # Generate unique task ID
        task_id = str(uuid_module.uuid4())
        
        # Set initial status in PostgreSQL Database
        create_task(task_id, "pending")
        
        # Start background task in separate thread
        thread = threading.Thread(
            target=run_calculation_generation_background,
            args=(task_id, session_id, pdf_content)
        )
        thread.start()
        
        logging.info(f"POLLING: Background task started with ID: {task_id}")
        
        return jsonify({"task_id": task_id}), 202
        
    except Exception as e:
        logging.error(f"POLLING: Critical error: {e}")
        import traceback
        logging.error(f"POLLING: Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/calculation_status/<task_id>', methods=['GET'])
def get_calculation_status(task_id):
    """Get the status of a calculation generation task"""
    try:
        # Retrieve status from PostgreSQL Database
        task_result = get_task_status(task_id)
        
        if not task_result:
            return jsonify({"status": "not_found"}), 404
        
        # Convert ObservedDict to regular dict for JSON serialization
        result_dict = dict(task_result)
        
        # If task is complete, update session storage
        if result_dict.get("status") == "complete" and result_dict.get("success"):
            session_id = session.get('session_id')
            if session_id:
                storage_manager = StorageManager()
                question_text = result_dict["data"]

                storage_manager.store_content(session_id, 'current_calculation_question', question_text)
                storage_manager.store_content(session_id, 'calculation_mode_active', True)

                try:
                    cleanup_task(task_id)
                except:
                    pass

        return jsonify(result_dict)

    except Exception as e:
        logging.error(f"Error checking calculation task status: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


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

    except Exception as e:
        logging.error(f"Error incrementing equation index: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/start_calculation_answer_check', methods=['POST'])
@csrf.exempt
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
        
        # Set context from stored content with fallback
        session_id = session.get('session_id')
        
        pdf_content = get_pdf_content_with_fallback()
        
        if not pdf_content:
            logging.info("No document content found for calculation answer checking")
            return jsonify({'error': 'No document content found'}), 400
        
        # Generate unique task ID
        task_id = str(uuid_module.uuid4())
        
        # Set initial status in PostgreSQL Database
        create_task(task_id, "pending")
        
        # Start background task in separate thread
        thread = threading.Thread(
            target=run_calculation_answer_check_background, 
            args=(task_id, challenge_question, user_answer, pdf_content)
        )
        thread.start()
        
        
        return jsonify({"task_id": task_id}), 202
        
    except Exception as e:
        logging.error(f"Critical error in calculation answer checking: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/calculation_answer_status/<task_id>', methods=['GET'])
def get_calculation_answer_status(task_id):
    """Get the status of a calculation answer check task"""
    try:
        # Retrieve status from PostgreSQL Database
        task_result = get_task_status(task_id)
        
        if not task_result:
            return jsonify({"status": "not_found"}), 404
        
        # Convert ObservedDict to regular dict for JSON serialization
        result_dict = dict(task_result)
        
        # If task is complete, also update session storage
        if result_dict.get("status") == "complete" and result_dict.get("success"):
            session_id = session.get('session_id')
            if session_id:
                storage_manager = StorageManager()
                evaluation_response = result_dict["data"]
                
                # Add evaluation to messages
                messages = storage_manager.retrieve_content(session_id, 'messages') or []
                messages.append({
                    "role": "assistant", 
                    "content": evaluation_response
                })
                storage_manager.store_content(session_id, 'messages', messages)
                
                # Clear any current calculation question
                storage_manager.delete_content(session_id, 'current_calculation_question')
                
                # Evaluation stored successfully
                
                # Clean up the task from database after successful completion
                try:
                    cleanup_task(task_id)
                except:
                    pass
        
        return jsonify(result_dict)
        
    except Exception as e:
        logging.error(f"Error checking calculation answer task status: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500



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
                    
        except Exception as e:
            logging.error(f"Document processing attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                return False, None, f"Error processing file after {max_retries} attempts: {str(e)}"
    
    return False, None, "Maximum retries exceeded"

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
                storage_manager.store_content(session_id, 'pdf_content', pdf_text)
                primary_storage.store_content(session_id, 'pdf_content', pdf_text)

                # Reset equation list so fresh notes start from equation 1
                storage_manager.store_content(session_id, 'equation_list', None)
                storage_manager.store_content(session_id, 'current_equation_index', 0)

                update_task_complete(task_id, success=True, data={
                    'success': True,
                    'filename': filename,
                    'message': f'Successfully loaded: {filename}',
                    'content_length': len(pdf_text) if pdf_text else 0
                })
                logging.info(f"Upload processing completed for task {task_id}")
            else:
                logging.error(f"Document processing failed for task {task_id}: {error_message}")
                update_task_failed(task_id, error_message or 'Document processing failed')
                
        except Exception as e:
            logging.error(f"Background file processing error for task {task_id}: {e}")
            update_task_failed(task_id, str(e))

@app.route('/upload', methods=['POST'])
@csrf.exempt
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
        from werkzeug.utils import secure_filename
        original_filename = file.filename
        secure_name = secure_filename(original_filename)
        
        if not secure_name:
            logging.warning(f"SECURITY: Invalid filename from session {session.get('session_id', 'unknown')}: {original_filename}")
            resource_monitor.increment('input_validation_failures')
            return jsonify({'error': 'Invalid filename'}), 400
        
        if file and secure_name and secure_name.lower().endswith(('.pdf', '.pptx')):
            
            # Check if there's already content in storage - if so, clear the session first
            session_id = session.get('session_id')
            if session_id:
                # Check both storage systems for existing content
                existing_content = (primary_storage.retrieve_content(session_id, 'pdf_content') or 
                                  storage_manager.retrieve_content(session_id, 'pdf_content'))
                if existing_content:
                    # Clear content from both storage systems
                    storage_manager.delete_content(session_id, 'pdf_content')
                    primary_storage.delete_content(session_id, 'pdf_content')
                    storage_manager.store_content(session_id, 'messages', [])
                    primary_storage.store_content(session_id, 'messages', [])
                    
            # Read file data into memory immediately
            file_data = file.read()
            
            # Generate unique task ID
            task_id = str(uuid_module.uuid4())
            
            # Initialize task status in database
            create_task(task_id, "pending")
            
            # Start background thread for file processing
            thread = threading.Thread(
                target=process_upload_background,
                args=(task_id, file_data, file.filename, session_id)
            )
            thread.daemon = True
            thread.start()
            
            # Store filename in session
            session['pdf_filename'] = file.filename
            
            # Clear previous messages (if they exist)
            try:
                storage_manager.store_content(session_id, 'messages', [])
            except:
                pass  # Ignore if session doesn't exist in storage yet
            
            try:
                primary_storage.store_content(session_id, 'messages', [])
            except:
                pass  # Ignore if session doesn't exist in storage yet
            
            # Return task_id for polling
            return jsonify({
                'task_id': task_id,
                'filename': file.filename
            })
        else:
            return jsonify({'error': 'Invalid file type. Please upload PDF or PPTX files only.'}), 400
            
    except Exception as e:
        logging.error(f"Critical error in file upload: {e}")
        return jsonify({'error': f'Critical upload error: {str(e)}'}), 500

@app.route('/upload_status/<task_id>', methods=['GET'])
def get_upload_status(task_id):
    """Get the status of a file upload task"""
    try:
        # Retrieve status from database
        task_result = get_task_status(task_id)
        
        if not task_result:
            return jsonify({"status": "not_found"}), 404
        
        # Convert to regular dict for JSON serialization
        result_dict = dict(task_result)
        
        # If task is complete, initialize TutorAI and set context
        if result_dict.get("status") == "complete" and result_dict.get("data", {}).get('success'):
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
        
    except Exception as e:
        logging.error(f"Error checking upload task status: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

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
            args=(task_id, user_message, pdf_content, conversation_history)
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
        
        # Convert ObservedDict to regular dict for JSON serialization
        def convert_observed(obj):
            """Recursively convert ObservedDict/ObservedList to regular dict/list"""
            # Check for ObservedDict first by checking class name
            if hasattr(obj, '__class__') and 'ObservedDict' in str(type(obj)):
                # Convert ObservedDict to regular dict
                return {key: convert_observed(value) for key, value in dict(obj).items()}
            elif hasattr(obj, '__class__') and 'ObservedList' in str(type(obj)):
                # Convert ObservedList to regular list
                return [convert_observed(item) for item in list(obj)]
            elif isinstance(obj, dict):
                return {key: convert_observed(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_observed(item) for item in obj]
            else:
                return obj
        
        result_dict = convert_observed(task_result)
        
        # Chat task result processing
        
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
        
    except Exception as e:
        logging.error(f"Error checking chat task status: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    init_session()
    
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'No message provided'}), 400
    
    user_message = data['message']
    
    # Add user message to file storage instead of session
    session_id = session.get('session_id')
    storage_manager = StorageManager()
    messages = storage_manager.retrieve_content(session_id, 'messages') or []
    messages.append({"role": "user", "content": user_message})
    storage_manager.store_content(session_id, 'messages', messages)
    
    try:
        # Removed equation selection mode - now using simplified direct calculation approach
        
        # Check if user is answering a calculation question
        if session_id:
            current_question = storage_manager.retrieve_content(session_id, 'current_calculation_question')
            
            if current_question:
                # Check if user wants to continue with more questions
                if user_message.strip().lower() == 'yes':
                    # For new questions, user should click the Calculation questions button
                    response_message = "To get a new calculation question, please click the **Calculation questions** button above. This will generate a fresh question using advanced mathematical reasoning."
                    
                    messages = storage_manager.retrieve_content(session_id, 'messages') or []
                    messages.append({
                        'role': 'assistant',
                        'content': response_message
                    })
                    storage_manager.store_content(session_id, 'messages', messages)
                    
                    return jsonify({
                        'success': True,
                        'response': response_message
                    })
                
                # Try to parse as a numerical answer
                try:
                    # Clean the answer to handle various formats
                    cleaned_answer = user_message.strip().replace('%', '').replace(',', '').replace('$', '').replace('£', '')
                    user_answer = float(cleaned_answer)
                    
                    # Return message to indicate async processing will start
                    return jsonify({
                        'success': True,
                        'response': f"I'm evaluating your answer: {user_message.strip()}. Please wait a moment for the detailed solution and feedback...",
                        'start_answer_check': True,
                        'challenge_question': current_question,
                        'user_answer': user_message.strip()
                    })
                except ValueError:
                    # Not a number, continue with normal chat processing
                    pass
        
        # Check if this is a simple word/phrase that doesn't need AI processing
        simple_words = user_message.strip().lower().split()
        if len(simple_words) <= 2 and all(word.isalpha() for word in simple_words):
            # For simple words, provide a quick response suggesting they elaborate
            quick_response = f"I see you mentioned '{user_message}'. Could you ask a more specific question about this topic from your lecture notes? For example, you could ask about definitions, calculations, examples, or concepts related to {user_message}."
            
            messages = storage_manager.retrieve_content(session_id, 'messages') or []
            messages.append({
                'role': 'assistant',
                'content': quick_response
            })
            storage_manager.store_content(session_id, 'messages', messages)
            
            return jsonify({
                'success': True,
                'response': quick_response
            })
        
        # For more complex messages, redirect to async polling approach
        return jsonify({
            'success': True,
            'response': f"Processing your message: '{user_message}'. Please wait while I analyze your lecture notes and provide a detailed response...",
            'start_async_chat': True,
            'user_message': user_message
        })
        
    except Exception as e:
        logging.error(f"Error in chat: {e}")
        error_msg = f"Error: {str(e)}"
        messages = storage_manager.retrieve_content(session_id, 'messages') or []
        messages.append({"role": "assistant", "content": error_msg})
        storage_manager.store_content(session_id, 'messages', messages)
        return jsonify({'error': error_msg}), 500

@app.route('/get_messages', methods=['GET'])
def get_messages():
    """DISABLED: This endpoint was causing content reloading issues"""
    logging.info("GET_MESSAGES: Endpoint disabled to prevent content reloading")
    return jsonify({'messages': []}), 200

@app.route('/start_quiz', methods=['POST'])
def start_quiz():
    """Legacy route - redirects to new async approach"""
    return start_quiz_generation()

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    """Submit quiz answer"""
    init_session()
    
    data = request.get_json()
    if not data or 'answer' not in data:
        logging.error("No answer provided in request data")
        return jsonify({'error': 'No answer provided'}), 400
    
    try:
        answer = data['answer']
        
        # Get quiz data from storage manager
        quiz_data = storage_manager.retrieve_content(session['session_id'], 'quiz_data')
        if not quiz_data:
            logging.error("No quiz data found in storage")
            return jsonify({'error': 'No quiz data found. Please start a new quiz.'}), 400
        
        questions = quiz_data.get('questions', [])
        current_index = quiz_data.get('current_question_index', 0)
        
        # Debug logging
        # Basic validation logging
        if not quiz_data.get('active', False):
            logging.info("Quiz submission received but quiz is not active")
        
        if not questions:
            logging.error("No quiz questions found in storage")
            return jsonify({'error': 'No quiz questions found. Please start a new quiz.'}), 400
            
        if not quiz_data.get('active', False):
            logging.error("Quiz is not active in storage")
            return jsonify({'error': 'Quiz is not active. Please start a new quiz.'}), 400
            
        if current_index >= len(questions):
            logging.error(f"Question index {current_index} out of bounds for {len(questions)} questions")
            return jsonify({'error': 'No more questions'}), 400
        
        current_question = questions[current_index]
        
        # Enhanced answer validation - normalize and compare
        user_answer = answer.strip()
        correct_answer = current_question['correct_answer'].strip()
        
        # Debug logging for answer comparison
        # Compare normalized answers
        logging.debug(f"  Options: {current_question.get('options', [])}")
        
        # Check if user answer exactly matches correct answer
        is_correct = user_answer == correct_answer
        
        # If no exact match, check if user selected an option that matches the correct answer
        if not is_correct and 'options' in current_question:
            options = current_question['options']
            # Check if user selected option text that matches correct answer
            for option in options:
                if user_answer == option.strip() and option.strip() == correct_answer:
                    is_correct = True
                    break
            
            # Additional fallback: check if user answer is an option that semantically matches
            if not is_correct:
                # Clean both answers for comparison (remove common prefixes)
                clean_user = user_answer.replace('Option A:', '').replace('Option B:', '').replace('Option C:', '').replace('Option D:', '').strip()
                clean_correct = correct_answer.replace('Option A:', '').replace('Option B:', '').replace('Option C:', '').replace('Option D:', '').strip()
                
                if clean_user == clean_correct:
                    is_correct = True
                    logging.debug(f"Match found after cleaning: '{clean_user}' == '{clean_correct}'")
        
        logging.debug(f"Final result: is_correct = {is_correct}")
        
        if is_correct:
            quiz_data['score'] += 1
        
        # Prepare explanation message
        raw_explanation = current_question.get('explanation', '')
        
        # Remove any existing "Correct!" variations from the beginning of the explanation to avoid duplication
        prefixes_to_remove = ['Correct!', 'Correct.', 'That\'s correct!', 'That is correct!', 'Right!', 'Yes!']
        for prefix in prefixes_to_remove:
            if raw_explanation.startswith(prefix):
                raw_explanation = raw_explanation[len(prefix):].strip()
                break
        
        # Add debug logging
        logging.debug(f"Raw explanation after cleanup: '{raw_explanation}'")
        
        if is_correct:
            explanation = f"✅ Correct! {raw_explanation}"
        else:
            explanation = f"❌ Incorrect. The correct answer is: {current_question['correct_answer']}. {raw_explanation}"
        
        logging.debug(f"Final explanation: '{explanation}'")
        
        # Update quiz data
        quiz_data['current_question_index'] += 1
        
        # Check if quiz is complete
        if quiz_data['current_question_index'] >= len(questions):
            quiz_data['active'] = False
            session['quiz_active'] = False
            
            # Store updated quiz data
            storage_manager.store_content(session['session_id'], 'quiz_data', quiz_data)
            
            return jsonify({
                'correct': is_correct,
                'explanation': explanation,
                'quiz_complete': True,
                'final_score': quiz_data['score'],
                'total_questions': len(questions)
            })
        
        # Store updated quiz data
        storage_manager.store_content(session['session_id'], 'quiz_data', quiz_data)
        
        # Return next question
        next_question = questions[quiz_data['current_question_index']]
        return jsonify({
            'correct': is_correct,
            'explanation': explanation,
            'next_question': next_question,
            'question_number': quiz_data['current_question_index'] + 1,
            'total_questions': len(questions),
            'quiz_complete': False
        })
        
    except Exception as e:
        logging.error(f"Error submitting answer: {e}")
        return jsonify({'error': f'Error submitting answer: {str(e)}'}), 500

# Removed old calculation handlers - now using simplified chat-based approach

@app.route('/start_essay_generation', methods=['POST'])
@csrf.exempt
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
        
        # Generate unique task ID
        task_id = str(uuid_module.uuid4())
        
        # Set initial status in PostgreSQL Database
        create_task(task_id, "pending")
        
        # Start background task in separate thread
        thread = threading.Thread(
            target=run_essay_generation_background, 
            args=(task_id, pdf_content)
        )
        thread.start()
        
        logging.info(f"ESSAY POLLING: Background task started with ID: {task_id}")
        
        return jsonify({"task_id": task_id}), 202
        
    except Exception as e:
        logging.error(f"ESSAY POLLING: Critical error: {e}")
        import traceback
        logging.error(f"ESSAY POLLING: Traceback: {traceback.format_exc()}")
        return jsonify({"error": "Failed to start essay generation"}), 500

@app.route('/essay_status/<task_id>', methods=['GET'])
def get_essay_status(task_id):
    """Get the status of an essay generation task"""
    try:
        # Retrieve status from PostgreSQL Database
        task_result = get_task_status(task_id)
        
        if not task_result:
            return jsonify({"status": "not_found"}), 404
        
        # Convert ObservedDict to regular dict for JSON serialization
        result_dict = dict(task_result)
        
        # If task is complete, also update session storage
        if result_dict.get("status") == "complete" and result_dict.get("success"):
            session_id = session.get('session_id')
            if session_id:
                storage_manager = StorageManager()
                essay_text = result_dict["data"]
                
                # Add essay to messages
                messages = storage_manager.retrieve_content(session_id, 'messages') or []
                messages.append({
                    "role": "assistant", 
                    "content": f"📝 **Essay Question:**\n\n{essay_text}"
                })
                storage_manager.store_content(session_id, 'messages', messages)
                
                logging.info(f"ESSAY POLLING: Essay stored in session for task {task_id}")
                
                # Clean up the task from database after successful completion
                try:
                    cleanup_task(task_id)
                except:
                    pass
        
        return jsonify(result_dict)
        
    except Exception as e:
        logging.error(f"ESSAY POLLING: Error getting task status for {task_id}: {e}")
        return jsonify({"error": "Failed to get task status"}), 500

@app.route('/start_key_concepts_generation', methods=['POST'])
@csrf.exempt
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
        
        # Generate unique task ID
        task_id = str(uuid_module.uuid4())
        
        # Set initial status in PostgreSQL Database
        create_task(task_id, "pending")
        
        # Start background task in separate thread
        thread = threading.Thread(
            target=run_key_concepts_generation_background, 
            args=(task_id, pdf_content)
        )
        thread.start()
        
        logging.info(f"KEY CONCEPTS POLLING: Background task started with ID: {task_id}")
        
        return jsonify({"task_id": task_id}), 202
        
    except Exception as e:
        logging.error(f"KEY CONCEPTS POLLING: Critical error: {e}")
        import traceback
        logging.error(f"KEY CONCEPTS POLLING: Traceback: {traceback.format_exc()}")
        return jsonify({"error": "Failed to start key concepts generation"}), 500

@app.route('/key_concepts_status/<task_id>', methods=['GET'])
def get_key_concepts_status(task_id):
    """Get the status of a key concepts explanation generation task"""
    try:
        # Retrieve status from PostgreSQL Database
        task_result = get_task_status(task_id)
        
        if not task_result:
            return jsonify({"status": "not_found"}), 404
        
        # Convert ObservedDict to regular dict for JSON serialization
        result_dict = dict(task_result)
        
        # If task is complete, also update session storage
        if result_dict.get("status") == "complete" and result_dict.get("success"):
            session_id = session.get('session_id')
            if session_id:
                storage_manager = StorageManager()
                key_concepts_text = result_dict["data"]
                
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
        
    except Exception as e:
        logging.error(f"KEY CONCEPTS POLLING: Error getting task status for {task_id}: {e}")
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
            'messages', 'quiz_questions', 'practice_equations'
        ]
        
        for content_type in content_types:
            try:
                storage_manager.delete_content(session_id, content_type)
            except Exception as e:
                logging.debug(f"Could not delete {content_type} for session {session_id}: {e}")
    
    # Clear Flask session
    session.clear()

@app.route('/clear_session', methods=['POST'])
@csrf.exempt
def clear_session():
    """Clear current session"""
    clear_session_data()
    return jsonify({'success': True})

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
