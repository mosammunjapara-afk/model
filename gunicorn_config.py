import os

bind = "0.0.0.0:" + str(os.environ.get("PORT", 10000))
workers = 1
worker_class = 'sync'
threads = 1

# CRITICAL: Increase timeouts
timeout = 600  # 10 minutes!
graceful_timeout = 600
keepalive = 5

# Restart workers frequently
max_requests = 50
max_requests_jitter = 5

# Preload for efficiency
preload_app = True

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Use shared memory
worker_tmp_dir = '/dev/shm'

def on_starting(server):
    print("🚀 Gunicorn is starting...")

def when_ready(server):
    print("✅ Gunicorn ready!")

def worker_abort(worker):
    print(f"❌ Worker {worker.pid} aborted!")
