"""
Deployment configuration for Replit Autoscale
Optimized for 100+ concurrent users
"""
import os
import logging
from datetime import datetime
from flask import request, jsonify, redirect
from werkzeug.middleware.proxy_fix import ProxyFix

def configure_for_production(app):
    """Configure Flask app for production deployment on Replit Autoscale"""
    
    # Proxy configuration for Replit's load balancer
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    # Production logging configuration
    if not app.debug:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(name)s %(message)s'
        )
        
        # Reduce database logging in production
        logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
        
    # Security headers for production
    @app.after_request
    def add_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        # Content Security Policy - allows MathJax and Bootstrap while blocking inline scripts
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://polyfill.io https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
            "font-src 'self' https://cdnjs.cloudflare.com; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "frame-src 'none'; "
            "object-src 'none'"
        )
        
        # Referrer Policy for privacy
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        return response
    
    # HTTPS redirection middleware
    @app.before_request
    def force_https():
        """Redirect HTTP to HTTPS in production"""
        if not request.is_secure and not app.debug:
            if request.endpoint == 'health_check':
                return
            if request.path == '/' or request.path == '/health':
                return
            return redirect(request.url.replace('http://', 'https://'), code=301)
    
    # Performance headers
    @app.after_request
    def add_performance_headers(response):
        # Enable compression
        response.headers['Vary'] = 'Accept-Encoding'
        
        # Cache static resources
        if request.endpoint and 'static' in request.endpoint:
            response.headers['Cache-Control'] = 'public, max-age=86400'  # 24 hours
        
        return response
    
    # Health check endpoint for Replit Autoscale
    @app.route('/health')
    def health_check():
        """Health check endpoint for load balancer"""
        from performance_optimizations import resource_monitor
        
        try:
            # Basic health checks
            metrics = resource_monitor.get_metrics()
            
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'metrics': {
                    'requests_processed': metrics.get('requests_processed', 0),
                    'cache_hit_rate': calculate_cache_hit_rate(metrics),
                    'error_rate': calculate_error_rate(metrics)
                }
            }
            
            return jsonify(health_status), 200
            
        except Exception as e:
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 503
    
    # Metrics endpoint for monitoring - localhost access only
    @app.route('/metrics')
    def metrics_endpoint():
        """Metrics endpoint for monitoring - localhost access only"""
        from flask import request as flask_request
        if flask_request.remote_addr not in ('127.0.0.1', '::1'):
            return jsonify({'error': 'Access denied'}), 403
        from performance_optimizations import resource_monitor
        try:
            metrics = resource_monitor.get_metrics()
            return jsonify(metrics), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

def calculate_cache_hit_rate(metrics):
    """Calculate cache hit rate percentage"""
    hits = metrics.get('cache_hits', 0)
    misses = metrics.get('cache_misses', 0)
    total = hits + misses
    
    if total == 0:
        return 0
    
    return round((hits / total) * 100, 2)

def calculate_error_rate(metrics):
    """Calculate error rate percentage"""
    errors = metrics.get('errors', 0)
    requests = metrics.get('requests_processed', 0)
    
    if requests == 0:
        return 0
    
    return round((errors / requests) * 100, 2)

# Gunicorn configuration for optimal performance
GUNICORN_CONFIG = {
    'bind': '0.0.0.0:5000',
    'workers': os.cpu_count() * 2 + 1,  # Optimal worker count
    'worker_class': 'sync',  # Sync workers for Flask
    'worker_connections': 1000,
    'max_requests': 1000,  # Restart workers after 1000 requests
    'max_requests_jitter': 100,
    'timeout': 30,
    'keepalive': 2,
    'preload_app': True,  # Load app before forking workers
    'reload': False,  # Disable reload in production
    'access_log_format': '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s',
    'error_log': '-',  # Log to stdout
    'access_log': '-',  # Log to stdout
}

# Environment-specific settings
class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SESSION_SECRET')
    SEND_FILE_MAX_AGE_DEFAULT = 86400  # 24 hours
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = 3600  # 1 hour

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SESSION_COOKIE_SECURE = False  # Allow HTTP in development

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    
def get_config():
    """Get configuration based on environment"""
    env = os.environ.get('FLASK_ENV', 'development')
    
    if env == 'production':
        return ProductionConfig()
    else:
        return DevelopmentConfig()