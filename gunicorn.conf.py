# Gunicorn configuration for Replit Autoscale deployment
# Optimized for 100+ concurrent users

import os
import multiprocessing

# Server socket
bind = "0.0.0.0:5000"
backlog = 2048

# Worker processes - Optimized for Replit with SSE streaming
# gthread allows multiple threads per worker so SSE streams don't block all requests
workers = min(multiprocessing.cpu_count() + 1, 4)  # Fewer workers, more threads each
worker_class = "gthread"  # Threaded workers required for SSE streaming
threads = 4  # Each worker handles 4 concurrent requests
worker_connections = 1000
max_requests = 2000
max_requests_jitter = 200

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
    """Called just after a worker has been forked"""
    server.log.info("Worker spawned (pid: %s)", worker.pid)

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
    """Called just after a worker has been exited — close all async HTTP clients"""
    try:
        from speed_optimizations import close_async_clients
        close_async_clients()
        server.log.info("Async clients closed for worker pid: %s", worker.pid)
    except Exception as e:
        server.log.warning("Error closing async clients on worker exit: %s", e)