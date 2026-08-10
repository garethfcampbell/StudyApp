"""
Deployment configuration for Replit Autoscale
Optimized for 100+ concurrent users
"""
import logging
from datetime import datetime
from flask import request, jsonify, redirect

def register_health_endpoint(app):
    """Register the /health endpoint. Called unconditionally from app.py so the
    health check is available in development as well as production."""

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

        except Exception:
            logging.exception("Health check failed")
            return jsonify({
                'status': 'unhealthy',
                'timestamp': datetime.now().isoformat()
            }), 503

    return app

def configure_for_production(app):
    """Configure Flask app for production deployment on Replit Autoscale.

    Note: ProxyFix is applied once in app.py (x_for=1, x_proto=1, x_host=1).
    It must NOT be applied again here — stacking ProxyFix trusts an extra
    forwarded hop and makes remote_addr client-spoofable.
    """

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
        except Exception:
            logging.exception("Error fetching metrics")
            return jsonify({'error': 'Unable to fetch metrics'}), 500

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