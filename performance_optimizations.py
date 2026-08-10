"""
Performance optimizations for Replit Autoscale deployment
Implements caching, connection pooling, and resource optimization
"""
import os
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Dict, Any
from flask import current_app
from models import SessionData

class SessionCache:
    """In-memory cache with TTL for frequently accessed session data"""
    
    def __init__(self, default_ttl: int = 600):  # Increased to 10 minutes for better caching
        self._cache = {}
        self._timestamps = {}
        self._lock = threading.RLock()
        self.default_ttl = default_ttl
        self.max_size = 20000  # Sized for 100+ concurrent users
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired"""
        with self._lock:
            if key not in self._cache:
                return None
                
            # Check if expired
            if time.time() - self._timestamps[key] > self.default_ttl:
                del self._cache[key]
                del self._timestamps[key]
                return None
                
            return self._cache[key]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache with TTL and size limit"""
        with self._lock:
            # Implement LRU eviction if cache is too large
            if len(self._cache) >= self.max_size:
                # Remove oldest 10% of entries
                sorted_items = sorted(self._timestamps.items(), key=lambda x: x[1])
                to_remove = sorted_items[:self.max_size // 10]
                for key_to_remove, _ in to_remove:
                    self._cache.pop(key_to_remove, None)
                    self._timestamps.pop(key_to_remove, None)
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
            
    def delete(self, key: str) -> None:
        """Delete item from cache"""
        with self._lock:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
            
    def clear_expired(self) -> int:
        """Clear all expired items and return count"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, timestamp in self._timestamps.items()
                if current_time - timestamp > self.default_ttl
            ]
            
            for key in expired_keys:
                del self._cache[key]
                del self._timestamps[key]
                
            return len(expired_keys)

class OptimizedStorageManager:
    """Enhanced storage manager with caching and batching"""
    
    def __init__(self):
        # Import here to avoid circular imports
        from database import postgres_db
        self.db = postgres_db
        self.cache = SessionCache()

    def _get_cache_key(self, session_id: str, content_type: str) -> str:
        """Generate cache key"""
        return f"cache_{session_id}_{content_type}"

    def _is_connection_error(self, error: Exception) -> bool:
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
        
    
    def store_content(self, session_id: str, content_type: str, content: Any) -> None:
        """Store content with caching and retry on connection errors"""
        cache_key = self._get_cache_key(session_id, content_type)

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
                # Update cache only after successful DB commit
                self.cache.set(cache_key, content)
                logging.debug(f"Stored optimized content for session {session_id}, type {content_type}")

        try:
            self._retry_db_operation(_do_store, operation_name="optimized_store")
        except Exception as e:
            # Invalidate cache on failure so stale data isn't served
            self.cache.delete(cache_key)
            logging.error(f"Error in optimized store after retries: {e}")
            raise
    
    def retrieve_content(self, session_id: str, content_type: str) -> Optional[Any]:
        """Retrieve content with cache-first approach and retry on connection errors"""
        cache_key = self._get_cache_key(session_id, content_type)
        cached_content = self.cache.get(cache_key)
        if cached_content is not None:
            return cached_content

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
                
                fetched = session_data.get_content()
                self.cache.set(cache_key, fetched)
                logging.debug(f"Retrieved optimized content for session {session_id}, type {content_type}")
                return fetched

        try:
            return self._retry_db_operation(_do_retrieve, operation_name="optimized_retrieve")
        except Exception as e:
            logging.error(f"Error in optimized retrieve after retries: {e}")
            return None
    
    def delete_content(self, session_id: str, content_type: str) -> None:
        """Delete content from both cache and database with retry"""
        cache_key = self._get_cache_key(session_id, content_type)
        self.cache.delete(cache_key)

        def _do_delete():
            with current_app.app_context():
                session_data = SessionData.query.filter_by(
                    session_id=session_id, 
                    content_type=content_type
                ).first()
                
                if session_data:
                    self.db.session.delete(session_data)
                    self.db.session.commit()

        try:
            self._retry_db_operation(_do_delete, operation_name="optimized_delete")
        except Exception as e:
            logging.error(f"Error in optimized delete after retries: {e}")
    
    def batch_store(self, operations: list) -> None:
        """Batch multiple store operations for efficiency"""
        try:
            for session_id, content_type, content in operations:
                self.store_content(session_id, content_type, content)
        except Exception as e:
            logging.error(f"Error in batch store: {e}")
    
    def batch_retrieve(self, session_id: str, content_types: list) -> Dict[str, Any]:
        """Retrieve multiple content types in a single database query for efficiency"""
        result = {}
        uncached_types = []
        
        # First, check cache for all requested types
        for content_type in content_types:
            cache_key = self._get_cache_key(session_id, content_type)
            cached_content = self.cache.get(cache_key)
            if cached_content is not None:
                result[content_type] = cached_content
            else:
                uncached_types.append(content_type)
        
        # If all items were in cache, return immediately
        if not uncached_types:
            logging.debug(f"Batch retrieve: all {len(content_types)} items from cache for session {session_id}")
            return result
        
        # Fetch uncached items from database in a single query
        try:
            with current_app.app_context():
                session_data_list = SessionData.query.filter(
                    SessionData.session_id == session_id,
                    SessionData.content_type.in_(uncached_types)
                ).all()
                
                for session_data in session_data_list:
                    if not session_data.is_expired():
                        content = session_data.get_content()
                        result[session_data.content_type] = content
                        # Cache the result
                        cache_key = self._get_cache_key(session_id, session_data.content_type)
                        self.cache.set(cache_key, content)
                    else:
                        # Clean up expired data
                        self.db.session.delete(session_data)
                
                self.db.session.commit()
                
        except Exception as e:
            logging.error(f"Error in batch retrieve: {e}")
        
        logging.debug(f"Batch retrieve: {len(result)} items for session {session_id} ({len(content_types) - len(uncached_types)} from cache, {len(uncached_types)} DB query)")
        return result
    
    def cleanup_cache(self) -> None:
        """Clean up expired cache entries"""
        expired_count = self.cache.clear_expired()
        logging.debug(f"Cleaned up {expired_count} expired cache entries")

def rate_limit(calls_per_minute: int = 60, use_session: bool = False):
    """Rate limiting with IP and session-based tracking.

    - The wrapped view is always invoked OUTSIDE the bookkeeping lock and at
      most ONCE per request; exceptions raised by the view propagate normally.
    - Internal rate-limiter errors fail open (the view still runs once).
    - Stale identifiers are pruned periodically so the tracking dict cannot
      grow without bound.
    """
    def decorator(func):
        # Per-endpoint storage for rate limiting data
        rate_limit_data = {}
        lock = threading.Lock()
        last_prune = [0.0]

        @wraps(func)
        def wrapper(*args, **kwargs):
            from flask import session, request, jsonify

            limited = False
            try:
                now = time.time()

                # Choose identifier: session ID if available and requested, otherwise IP
                if use_session and 'session_id' in session:
                    identifier = session.get('session_id')
                    key_type = "session"
                else:
                    # remote_addr is proxy-corrected by ProxyFix (x_for=1) in app.py
                    identifier = request.remote_addr or 'unknown'
                    key_type = "IP"

                key = f"{key_type}_{identifier}"

                with lock:
                    # Periodically prune identifiers that have gone quiet
                    if now - last_prune[0] > 300:
                        stale_keys = [k for k, times in rate_limit_data.items()
                                      if not times or now - times[-1] > 60]
                        for stale_key in stale_keys:
                            rate_limit_data.pop(stale_key, None)
                        last_prune[0] = now

                    calls = rate_limit_data.setdefault(key, [])

                    # Remove calls older than 1 minute
                    calls[:] = [call_time for call_time in calls if now - call_time < 60]

                    if len(calls) >= calls_per_minute:
                        limited = True
                    else:
                        calls.append(now)

                if limited:
                    resource_monitor.increment('security_violations')
                    logging.warning(f"SECURITY: Rate limit exceeded for {key_type} {identifier}: "
                                    f"{calls_per_minute}/min on endpoint {func.__name__} "
                                    f"(path {request.path}, UA {request.headers.get('User-Agent', 'Unknown')})")
                    return jsonify({
                        'error': 'Rate limit exceeded',
                        'retry_after': 60,
                        'message': 'Too many requests. Please wait before trying again.'
                    }), 429

            except Exception as e:
                # Rate-limiter internal failure: fail open (view still called once below)
                logging.error(f"Rate limiting error in {func.__name__}: {e}")

            # Invoke the view exactly once, outside the lock; its exceptions propagate
            return func(*args, **kwargs)

        return wrapper
    return decorator

class ResourceMonitor:
    """Monitor resource usage and performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'requests_processed': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'db_operations': 0,
            'ai_calls': 0,
            'errors': 0,
            'security_violations': 0,
            'input_validation_failures': 0,
            'csrf_failures': 0
        }
        self._lock = threading.Lock()
    
    def increment(self, metric: str, value: int = 1):
        """Increment a metric counter"""
        with self._lock:
            self.metrics[metric] = self.metrics.get(metric, 0) + value
    
    def get_metrics(self) -> Dict[str, int]:
        """Get current metrics"""
        with self._lock:
            return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset all metrics"""
        with self._lock:
            for key in self.metrics:
                self.metrics[key] = 0

# Global instances
optimized_storage = OptimizedStorageManager()
resource_monitor = ResourceMonitor()

def invalidate_optimized_cache(session_id: str, content_type: str) -> None:
    """Invalidate the in-memory cache entry for (session_id, content_type).

    Called by DatabaseStorageManager after direct DB writes so the optimized
    cache never serves stale data for content written through the other path.
    """
    try:
        cache_key = optimized_storage._get_cache_key(session_id, content_type)
        optimized_storage.cache.delete(cache_key)
    except Exception as e:
        logging.debug(f"Cache invalidation failed for {session_id}/{content_type}: {e}")

# Cleanup task that runs periodically
def periodic_cleanup():
    """Periodic cleanup task for maintenance — clears in-memory cache"""
    try:
        optimized_storage.cleanup_cache()
        logging.debug("Periodic cache cleanup completed")
    except Exception as e:
        logging.error(f"Error in periodic cleanup: {e}")

def periodic_db_cleanup():
    """Periodic cleanup of expired database sessions — runs every 30 minutes"""
    try:
        from app import app
        from database_storage_manager import DatabaseStorageManager
        with app.app_context():
            db_manager = DatabaseStorageManager()
            db_manager.cleanup_expired_sessions()
    except Exception as e:
        logging.error(f"Error in DB session cleanup: {e}")

_cleanup_started = False
_cleanup_start_lock = threading.Lock()

def start_periodic_cleanup():
    """Start the periodic cleanup threads (idempotent within a process).

    Called at import time in app.py so `python main.py` (dev) gets cleanup
    threads. Under gunicorn with preload_app the import-time threads live in
    the master and do NOT survive fork, so gunicorn's post_fork hook calls
    restart_periodic_cleanup_after_fork() to start them in each worker.
    """
    global _cleanup_started
    with _cleanup_start_lock:
        if _cleanup_started:
            return
        _cleanup_started = True

    def cache_cleanup_loop():
        while True:
            time.sleep(300)  # 5 minutes
            periodic_cleanup()

    def db_cleanup_loop():
        while True:
            time.sleep(1800)  # 30 minutes
            periodic_db_cleanup()

    cache_thread = threading.Thread(target=cache_cleanup_loop, daemon=True)
    cache_thread.start()
    db_thread = threading.Thread(target=db_cleanup_loop, daemon=True)
    db_thread.start()

def restart_periodic_cleanup_after_fork():
    """Reset the started flag and start cleanup threads in a forked worker.

    Threads never survive fork(), so the flag inherited from the master is
    stale; clear it and start fresh threads in this process.
    """
    global _cleanup_started
    with _cleanup_start_lock:
        _cleanup_started = False
    start_periodic_cleanup()