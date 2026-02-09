import multiprocessing
import os

# Binding
bind = "0.0.0.0:" + str(os.environ.get("PORT", 10000))

# Worker processes
workers = 1
worker_class = 'sync'
threads = 1

# Timeouts
timeout = 300
graceful_timeout = 300
keepalive = 5

# Restart workers
max_requests = 100
max_requests_jitter = 10

# Preload app for memory efficiency
preload_app = True

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Worker memory limits
worker_tmp_dir = '/dev/shm'

def on_starting(server):
    print("🚀 Gunicorn server is starting...")

def on_reload(server):
    print("🔄 Gunicorn server is reloading...")

def when_ready(server):
    print("✅ Gunicorn server is ready. Spawning workers...")

def worker_int(worker):
    print(f"⚠️ Worker received INT or QUIT signal: {worker.pid}")

def worker_abort(worker):
    print(f"❌ Worker received SIGABRT signal: {worker.pid}")
