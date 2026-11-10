"""Gunicorn settings. Override with env vars: PORT, WEB_CONCURRENCY, GUNICORN_TIMEOUT."""
import multiprocessing
import os

bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", min(multiprocessing.cpu_count() * 2 + 1, 4)))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "30"))
accesslog = "-"  # access log to stdout
errorlog = "-"
loglevel = os.environ.get("LOG_LEVEL", "info").lower()
