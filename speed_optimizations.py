"""
Additional speed optimizations for AI Tutor
Implements response caching, connection pooling, and async improvements
"""
import asyncio
import aiohttp
import hashlib
import json
import logging
from functools import wraps
from typing import Dict, Any, Optional
from performance_optimizations import SessionCache

# Global response cache for AI responses
response_cache = SessionCache(default_ttl=1800)  # 30-minute cache for AI responses

# Connection pool for HTTP requests
HTTP_CONNECTOR = None

def get_http_connector():
    """Get or create HTTP connector with optimized settings"""
    global HTTP_CONNECTOR
    if HTTP_CONNECTOR is None:
        HTTP_CONNECTOR = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=300,  # Keep connections alive longer
            enable_cleanup_closed=True
        )
    return HTTP_CONNECTOR

def cache_ai_response(cache_key_prefix: str = "ai_response"):
    """Decorator to cache AI responses based on content hash"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            cache_content = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            cache_key = f"{cache_key_prefix}_{hashlib.md5(cache_content.encode()).hexdigest()}"
            
            # Try to get from cache first
            cached_result = response_cache.get(cache_key)
            if cached_result:
                logging.info(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            if result:  # Only cache successful results
                response_cache.set(cache_key, result)
                logging.info(f"Cached result for {func.__name__}")
            
            return result
        return wrapper
    return decorator

class FastAIClient:
    """Optimized AI client with connection pooling and caching"""
    
    def __init__(self):
        self.session = None
        self._lock = asyncio.Lock()
    
    async def get_session(self):
        """Get or create aiohttp session with optimized settings"""
        if self.session is None or self.session.closed:
            async with self._lock:
                if self.session is None or self.session.closed:
                    connector = get_http_connector()
                    timeout = aiohttp.ClientTimeout(
                        total=120,  # Total timeout
                        connect=10,  # Connection timeout
                        sock_read=60  # Socket read timeout
                    )
                    self.session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers={
                            'Connection': 'keep-alive',
                            'Keep-Alive': 'timeout=300, max=1000'
                        }
                    )
        return self.session
    
    @cache_ai_response("gemini_response")
    async def call_gemini_cached(self, prompt: str, context: str = "") -> Optional[str]:
        """Cached Gemini API call"""
        # This would integrate with your existing Gemini client
        # Implementation would depend on your current TutorAI class
        pass
    
    @cache_ai_response("openai_response") 
    async def call_openai_cached(self, prompt: str, context: str = "") -> Optional[str]:
        """Cached OpenAI API call"""
        # This would integrate with your existing OpenAI client
        # Implementation would depend on your current TutorAI class
        pass

# Global instance
fast_ai_client = FastAIClient()

async def _close_async_resources():
    """Close all open aiohttp sessions and connectors"""
    global HTTP_CONNECTOR
    try:
        if fast_ai_client.session and not fast_ai_client.session.closed:
            await fast_ai_client.session.close()
            fast_ai_client.session = None
            logging.info("FastAIClient session closed")
    except Exception as e:
        logging.warning(f"Error closing FastAIClient session: {e}")
    try:
        if HTTP_CONNECTOR and not HTTP_CONNECTOR.closed:
            await HTTP_CONNECTOR.close()
            HTTP_CONNECTOR = None
            logging.info("HTTP connector closed")
    except Exception as e:
        logging.warning(f"Error closing HTTP connector: {e}")

def close_async_clients():
    """Synchronously close all async HTTP clients — safe to call from gunicorn worker_exit"""
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_close_async_resources())
        loop.close()
    except Exception as e:
        logging.warning(f"Error during async client cleanup: {e}")

def preload_critical_resources():
    """Preload frequently used resources into cache"""
    logging.info("Preloading critical resources for faster responses...")
    
    # Pre-cache common prompt templates
    common_prompts = [
        "explain_key_concepts",
        "generate_quiz_questions", 
        "create_summary",
        "calculation_help"
    ]
    
    for prompt_type in common_prompts:
        cache_key = f"prompt_template_{prompt_type}"
        # This would cache your prompt templates
        # response_cache.set(cache_key, get_prompt_template(prompt_type))
    
    logging.info("Critical resources preloaded")

def optimize_json_responses(response_data: Dict[Any, Any]) -> str:
    """Optimize JSON serialization for faster responses"""
    return json.dumps(
        response_data,
        separators=(',', ':'),  # Compact JSON
        ensure_ascii=False,     # Allow unicode
        sort_keys=False         # Don't sort for speed
    )

class ResponseCompressor:
    """Compress responses for faster transfer"""
    
    @staticmethod
    def should_compress(content: str, min_size: int = 1000) -> bool:
        """Check if content should be compressed"""
        return len(content.encode('utf-8')) > min_size
    
    @staticmethod
    def compress_response(content: str) -> bytes:
        """Compress response content"""
        import gzip
        return gzip.compress(content.encode('utf-8'))

# Performance monitoring
class SpeedMetrics:
    """Track response times and optimization effectiveness"""
    
    def __init__(self):
        self.response_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
    def record_response_time(self, time_ms: float):
        """Record response time"""
        self.response_times.append(time_ms)
        # Keep only last 1000 measurements
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
    
    def get_average_response_time(self) -> float:
        """Get average response time"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)
    
    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate percentage"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100

# Global metrics instance
speed_metrics = SpeedMetrics()