import json
import uuid
import logging
import time
from datetime import datetime, timedelta
from flask import current_app
from models import SessionData
from database import postgres_db

class DatabaseStorageManager:
    """Database-based storage manager using PostgreSQL"""
    
    def __init__(self):
        self.db = postgres_db

    def _is_connection_error(self, error):
        error_str = str(error).lower()
        connection_keywords = ['ssl', 'connection', 'eof detected', 'closed unexpectedly',
                              'broken pipe', 'timeout', 'operationalerror', 'connection refused']
        return any(kw in error_str for kw in connection_keywords)

    def _recover_connection(self):
        try:
            self.db.session.rollback()
        except Exception:
            pass
        try:
            self.db.session.remove()
        except Exception:
            pass

    def _retry_db_operation(self, operation, max_retries=3, operation_name="db_operation"):
        last_error = None
        for attempt in range(max_retries):
            try:
                return operation()
            except Exception as e:
                last_error = e
                if self._is_connection_error(e) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    logging.warning(f"Connection error in {operation_name} (attempt {attempt + 1}/{max_retries}), "
                                   f"recovering in {wait_time}s: {e}")
                    self._recover_connection()
                    time.sleep(wait_time)
                elif attempt < max_retries - 1:
                    try:
                        self.db.session.rollback()
                    except Exception:
                        pass
                    time.sleep(1)
                else:
                    try:
                        self.db.session.rollback()
                    except Exception:
                        pass
                    raise last_error
        
    def generate_session_id(self):
        return str(uuid.uuid4())
    
    def store_content(self, session_id, content_type, content):
        """Store content for a session in database with retry"""
        def _do_store():
            with current_app.app_context():
                existing = SessionData.query.filter_by(
                    session_id=session_id, 
                    content_type=content_type
                ).first()
                
                if existing:
                    existing.set_content(content)
                    existing.timestamp = datetime.utcnow()
                    existing.expires_at = datetime.utcnow() + timedelta(hours=24)
                else:
                    session_data = SessionData(
                        session_id=session_id,
                        content_type=content_type,
                        content=content
                    )
                    self.db.session.add(session_data)
                
                self.db.session.commit()
                logging.debug(f"Stored content in DB for session {session_id}, type {content_type}")

        try:
            self._retry_db_operation(_do_store, operation_name="db_store")
        except Exception as e:
            logging.error(f"Error storing content in DB after retries: {e}")
            raise Exception(f"Failed to store content: {str(e)}")
    
    def retrieve_content(self, session_id, content_type):
        """Retrieve content for a session from database with retry"""
        def _do_retrieve():
            with current_app.app_context():
                session_data = SessionData.query.filter_by(
                    session_id=session_id, 
                    content_type=content_type
                ).first()
                
                if not session_data:
                    return None
                
                if session_data.is_expired():
                    self.delete_content(session_id, content_type)
                    return None
                
                logging.debug(f"Retrieved content from DB for session {session_id}, type {content_type}")
                return session_data.get_content()

        try:
            return self._retry_db_operation(_do_retrieve, operation_name="db_retrieve")
        except Exception as e:
            logging.error(f"Error retrieving content from DB after retries: {e}")
            return None
    
    def delete_content(self, session_id, content_type):
        """Delete content for a session from database with retry"""
        def _do_delete():
            with current_app.app_context():
                session_data = SessionData.query.filter_by(
                    session_id=session_id, 
                    content_type=content_type
                ).first()
                
                if session_data:
                    self.db.session.delete(session_data)
                    self.db.session.commit()
                    logging.debug(f"Deleted content from DB for session {session_id}, type {content_type}")

        try:
            self._retry_db_operation(_do_delete, operation_name="db_delete")
        except Exception as e:
            logging.error(f"Error deleting content from DB after retries: {e}")
    
    def clear_session(self, session_id):
        """Clear all content for a session"""
        try:
            with current_app.app_context():
                # Delete all session data for this session
                deleted_count = SessionData.query.filter_by(session_id=session_id).delete()
                self.db.session.commit()
                
                logging.debug(f"Cleared all content for session {session_id} (deleted {deleted_count} records)")
            
        except Exception as e:
            self.db.session.rollback()
            logging.error(f"Error clearing session {session_id}: {e}")
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions (optional maintenance function)"""
        try:
            with current_app.app_context():
                current_time = datetime.utcnow()
                
                # Delete all expired session data
                deleted_count = SessionData.query.filter(
                    SessionData.expires_at < current_time
                ).delete()
                
                self.db.session.commit()
                logging.info(f"Cleaned up {deleted_count} expired session entries")
            
        except Exception as e:
            self.db.session.rollback()
            logging.error(f"Error during session cleanup: {e}")