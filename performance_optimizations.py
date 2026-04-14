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
from functools import lru_cache, wraps
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
        self._batch_operations = []
        self._batch_lock = threading.Lock()
        
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

class ConnectionPool:
    """Simple connection pool for AI clients"""
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self._pool = []
        self._lock = threading.Lock()
        self._created_count = 0
    
    def get_client(self, client_factory):
        """Get a client from pool or create new one"""
        with self._lock:
            if self._pool:
                return self._pool.pop()
            elif self._created_count < self.max_size:
                client = client_factory()
                self._created_count += 1
                return client
            else:
                # Pool is full, return a new temporary client
                return client_factory()
    
    def return_client(self, client):
        """Return client to pool"""
        with self._lock:
            if len(self._pool) < self.max_size:
                self._pool.append(client)

def rate_limit(calls_per_minute: int = 60, use_session: bool = False):
    """Enhanced rate limiting with IP and session-based tracking"""
    def decorator(func):
        # Global storage for rate limiting data
        rate_limit_data = {}
        lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                from flask import session, request, jsonify
                
                now = time.time()
                
                # Choose identifier: session ID if available and requested, otherwise IP
                if use_session and 'session_id' in session:
                    identifier = session.get('session_id')
                    key_type = "session"
                    logging.debug(f"Rate limiting by session: {identifier}")
                else:
                    # Simple IP extraction
                    identifier = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
                    if ',' in identifier:
                        identifier = identifier.split(',')[0].strip()
                    key_type = "IP"
                    logging.debug(f"Rate limiting by IP: {identifier}")
                
                key = f"{key_type}_{identifier}"
                
                with lock:
                    # Get existing call data for this identifier
                    if key not in rate_limit_data:
                        rate_limit_data[key] = []
                    
                    calls = rate_limit_data[key]
                    
                    # Remove calls older than 1 minute
                    calls[:] = [call_time for call_time in calls if now - call_time < 60]
                    
                    if len(calls) >= calls_per_minute:
                        # Enhanced security logging for rate limit violations
                        logging.warning(f"SECURITY: Rate limit exceeded for {key_type} {identifier}: {len(calls)} requests in last minute for endpoint {func.__name__}")
                        # Log additional context
                        logging.warning(f"SECURITY: User-Agent: {request.headers.get('User-Agent', 'Unknown')}")
                        logging.warning(f"SECURITY: Request path: {request.path}")
                        
                        if request.is_json and hasattr(request, 'get_json'):
                            try:
                                data = request.get_json(silent=True)
                                if data:
                                    logging.warning(f"SECURITY: Request contained JSON data: {type(data)}")
                            except:
                                pass
                        
                        raise Exception(f"Rate limit exceeded: {calls_per_minute} calls per minute")
                    
                    # Add current request
                    calls.append(now)
                    rate_limit_data[key] = calls
                    
                    logging.debug(f"Rate limit check passed: {len(calls)}/{calls_per_minute} requests for {identifier}")
                    return func(*args, **kwargs)
                    
            except Exception as e:
                # Log security violations
                if "Rate limit exceeded" in str(e):
                    resource_monitor.increment('security_violations')
                    logging.error(f"SECURITY: Rate limit violation - {e}")
                    return jsonify({
                        'error': 'Rate limit exceeded',
                        'retry_after': 60,
                        'message': 'Too many requests. Please wait before trying again.'
                    }), 429
                else:
                    logging.error(f"Rate limiting error: {e}")
                    # Continue with the original function if rate limiting fails
                    return func(*args, **kwargs)
        
        return wrapper
    return decorator

@lru_cache(maxsize=128)
def cached_pdf_processing(content_hash: str, content: str) -> str:
    """Cache PDF processing results"""
    # This would be called with a hash of the PDF content
    # to avoid reprocessing the same document
    return content  # Simplified - actual processing would happen here

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
ai_connection_pool = ConnectionPool(max_size=5)

# Cleanup task that runs periodically
def periodic_cleanup():
    """Periodic cleanup task for maintenance"""
    try:
        optimized_storage.cleanup_cache()
        logging.debug("Periodic cleanup completed")
    except Exception as e:
        logging.error(f"Error in periodic cleanup: {e}")

# Schedule cleanup every 5 minutes
import threading
def start_periodic_cleanup():
    """Start the periodic cleanup thread"""
    def cleanup_loop():
        while True:
            time.sleep(300)  # 5 minutes
            periodic_cleanup()
    
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()