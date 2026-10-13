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

    @staticmethod
    def _invalidate_optimized_cache(session_id, content_type):
        """Keep OptimizedStorageManager's in-memory cache coherent with direct
        DB writes made through this manager. Lazy import avoids circularity."""
        try:
            from performance_optimizations import invalidate_optimized_cache
            invalidate_optimized_cache(session_id, content_type)
        except Exception as e:
            logging.debug(f"Optimized cache invalidation skipped: {e}")

    def store_content(self, session_id, content_type, content):
        """Store content for a session in database with retry.

        One UPDATE statement when the key already exists (the common case);
        only a brand-new key costs a second statement (INSERT). This used to be
        a SELECT followed by an UPDATE or INSERT on every write."""
        def _do_store():
            with current_app.app_context():
                now = datetime.utcnow()
                updated = SessionData.query.filter_by(
                    session_id=session_id, content_type=content_type
                ).update({
                    SessionData.content: json.dumps(content, ensure_ascii=False),
                    SessionData.timestamp: now,
                    SessionData.expires_at: now + timedelta(hours=24),
                }, synchronize_session=False)
                if not updated:
                    self.db.session.add(SessionData(
                        session_id=session_id, content_type=content_type, content=content))
                self.db.session.commit()
                logging.debug(f"Stored content in DB for session {session_id}, type {content_type}")

        try:
            self._retry_db_operation(_do_store, operation_name="db_store")
            # Invalidate the optimized in-memory cache so readers don't see stale data
            self._invalidate_optimized_cache(session_id, content_type)
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
    
    def batch_retrieve(self, session_id, content_types):
        """Read several content types for a session in ONE query (no in-memory
        cache, so the result is always current across gunicorn workers).
        Returns {content_type: value} for the types that exist and are unexpired."""
        def _do_batch():
            with current_app.app_context():
                rows = SessionData.query.filter(
                    SessionData.session_id == session_id,
                    SessionData.content_type.in_(list(content_types))
                ).all()
                out = {}
                for row in rows:
                    if row.is_expired():
                        continue
                    out[row.content_type] = row.get_content()
                return out

        try:
            return self._retry_db_operation(_do_batch, operation_name="db_batch_retrieve") or {}
        except Exception as e:
            logging.error(f"Error batch-retrieving content from DB after retries: {e}")
            return {}

    def batch_store(self, operations):
        """Store several (session_id, content_type, content) triples with ONE
        SELECT of the existing rows, in-place updates / inserts, and ONE commit."""
        ops = list(operations)
        if not ops:
            return

        def _do_batch_store():
            with current_app.app_context():
                now = datetime.utcnow()
                sids = {sid for sid, _, _ in ops}
                types = {ct for _, ct, _ in ops}
                rows = SessionData.query.filter(
                    SessionData.session_id.in_(list(sids)),
                    SessionData.content_type.in_(list(types))
                ).all()
                existing = {(r.session_id, r.content_type): r for r in rows}
                for sid, ct, content in ops:
                    row = existing.get((sid, ct))
                    if row:
                        row.set_content(content)
                        row.timestamp = now
                        row.expires_at = now + timedelta(hours=24)
                    else:
                        row = SessionData(session_id=sid, content_type=ct, content=content)
                        self.db.session.add(row)
                        existing[(sid, ct)] = row
                self.db.session.commit()

        try:
            self._retry_db_operation(_do_batch_store, operation_name="db_batch_store")
            for sid, ct, _ in ops:
                self._invalidate_optimized_cache(sid, ct)
        except Exception as e:
            logging.error(f"Error batch-storing content in DB after retries: {e}")
            raise Exception(f"Failed to store content: {str(e)}")

    def delete_content(self, session_id, content_type):
        """Delete content for a session from database with retry (one statement)"""
        def _do_delete():
            with current_app.app_context():
                n = SessionData.query.filter_by(
                    session_id=session_id, content_type=content_type
                ).delete(synchronize_session=False)
                self.db.session.commit()
                if n:
                    logging.debug(f"Deleted content from DB for session {session_id}, type {content_type}")

        try:
            self._retry_db_operation(_do_delete, operation_name="db_delete")
            self._invalidate_optimized_cache(session_id, content_type)
        except Exception as e:
            logging.error(f"Error deleting content from DB after retries: {e}")
    
    def delete_many(self, session_id, content_types):
        """Delete several content types for a session in ONE statement (instead of
        one round trip per type) and keep the in-memory cache coherent."""
        types = list(content_types)

        def _do_delete_many():
            with current_app.app_context():
                n = SessionData.query.filter(
                    SessionData.session_id == session_id,
                    SessionData.content_type.in_(types)
                ).delete(synchronize_session=False)
                self.db.session.commit()
                return n

        try:
            deleted = self._retry_db_operation(_do_delete_many, operation_name="db_delete_many")
            for t in types:
                self._invalidate_optimized_cache(session_id, t)
            logging.debug(f"Deleted {deleted} row(s) for session {session_id} in one statement")
            return deleted
        except Exception as e:
            logging.error(f"Error bulk-deleting content from DB after retries: {e}")
            return 0

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