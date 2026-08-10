# Gunicorn configuration for Replit Autoscale deployment
# Optimized for 100+ concurrent users

import os
import multiprocessing

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes - Optimized for 100+ concurrent users with SSE streaming
# gthread allows multiple threads per worker so SSE streams don't block all requests
workers = min(multiprocessing.cpu_count(), 6)  # Match CPU count for max throughput
worker_class = "gthread"  # Threaded workers required for SSE streaming
threads = 12  # Each worker handles 12 concurrent requests (6 workers × 12 threads = 72 slots)
worker_connections = 2000
max_requests = 3000
max_requests_jitter = 300

# Timeouts - Optimized for faster responses
timeout = 300  # Increased for file uploads and AI processing
keepalive = 5  # Increased keepalive for connection reuse
worker_tmp_dir = "/dev/shm"  # Use shared memory for better performance

# Process naming
proc_name = 'ai-tutor'

# Server mechanics
preload_app = True
reload = False  # Disable in production

# Logging
accesslog = '-'  # Log to stdout
errorlog = '-'   # Log to stderr
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process management
pidfile = '/tmp/gunicorn.pid'
user = None
group = None
tmp_upload_dir = None

# SSL (not needed on Replit - handled by load balancer)
keyfile = None
certfile = None

# Environment variables
raw_env = [
    'FLASK_ENV=production',
]

def post_fork(server, worker):
    """Called just after a worker has been forked.

    With preload_app=True the SQLAlchemy engine is created in the master
    before forking; forked workers would otherwise share the master's pooled
    DB sockets, which is the root cause of intermittent SSL-EOF errors.
    Dispose the inherited pool so each worker builds fresh connections.
    """
    server.log.info("Worker spawned (pid: %s)", worker.pid)
    try:
        from app import app
        from database import postgres_db
        with app.app_context():
            try:
                # SQLAlchemy >= 1.4.33: discard inherited connections without
                # closing them (they still belong to the master process)
                postgres_db.engine.dispose(close=False)
            except TypeError:
                postgres_db.engine.dispose()
        server.log.info("SQLAlchemy engine disposed in worker (pid: %s)", worker.pid)
    except Exception as e:
        server.log.warning("Could not dispose SQLAlchemy engine in worker: %s", e)

    # Cleanup threads started at import time live in the master and do not
    # survive fork — restart them inside each worker.
    try:
        from performance_optimizations import restart_periodic_cleanup_after_fork
        restart_periodic_cleanup_after_fork()
    except Exception as e:
        server.log.warning("Could not restart periodic cleanup in worker: %s", e)

def pre_fork(server, worker):
    """Called just before a worker is forked"""
    pass

def when_ready(server):
    """Called just after the server is started"""
    server.log.info("Server is ready. Spawning workers")

def worker_int(worker):
    """Called just after a worker received the SIGINT or SIGQUIT signal"""
    worker.log.info("worker received INT or QUIT signal")

def pre_exec(server):
    """Called just before a new master process is forked"""
    server.log.info("Forked child, re-executing.")

def post_worker_init(worker):
    """Called just after a worker has initialized the application"""
    worker.log.info("Worker initialized")

def worker_exit(server, worker):
    """Called just after a worker has exited.

    No explicit async-client cleanup is required: TutorAI manages its own
    client lifecycle (close_async_clients() is a safe no-op) and the process
    teardown releases any remaining sockets.
    """
    server.log.info("Worker exited (pid: %s)", worker.pid)