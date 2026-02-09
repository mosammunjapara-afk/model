import multiprocessing
import os

# Binding
bind = "0.0.0.0:" + str(os.environ.get("PORT", 10000))

# Worker processes
workers = 1  # Keep it 1 for low memory
worker_class = 'sync'
threads = 1

# Timeouts
timeout = 300  # 5 minutes for model inference
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
worker_tmp_dir = '/dev/shm'  # Use shared memory for better performance

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
```

## **3. `Procfile` (New File)**
```
web: gunicorn --config gunicorn_config.py app:app
```

## **4. `requirements.txt` (Updated)**
```
Flask==3.0.0
tensorflow==2.16.1
numpy==1.26.3
Pillow==10.2.0
gunicorn==21.2.0
h5py==3.10.0
