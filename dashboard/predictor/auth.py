"""
auth.py
=======
MySQL-credential-based authentication helpers for the predictor app.

Security measures:
  - Credentials validated by attempting a real MySQL connection (pymysql).
  - Passwords are NEVER stored in the session; only a boolean flag is kept.
  - Failed login attempts are rate-limited per IP (in-memory, thread-safe).
  - Sessions expire after SESSION_COOKIE_AGE (configured in settings.py).
  - All protected views require the session flag to be set.
"""

import time
import threading
import pymysql
from functools import wraps
from django.shortcuts import redirect
from django.http import JsonResponse
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import data_layer

# ── Rate limiter (per-IP, in-memory, thread-safe) ──────────────────────────

_lock = threading.Lock()
_attempts = {}  # ip -> [timestamp, ...]

MAX_ATTEMPTS = 5  # max failed attempts per window
WINDOW_SECONDS = 300  # 5-minute window
LOCKOUT_SECONDS = 300  # 5-minute lockout after exceeding limit


def _clean_old_attempts(ip, now):
    """Remove attempts older than the window."""
    if ip in _attempts:
        _attempts[ip] = [t for t in _attempts[ip] if now - t < WINDOW_SECONDS]
        if not _attempts[ip]:
            del _attempts[ip]


def is_rate_limited(ip):
    """Check if an IP is currently rate-limited."""
    now = time.time()
    with _lock:
        _clean_old_attempts(ip, now)
        if ip in _attempts and len(_attempts[ip]) >= MAX_ATTEMPTS:
            oldest = _attempts[ip][0]
            if now - oldest < LOCKOUT_SECONDS:
                return True
    return False


def record_failed_attempt(ip):
    """Record a failed login attempt for the given IP."""
    now = time.time()
    with _lock:
        _clean_old_attempts(ip, now)
        _attempts.setdefault(ip, []).append(now)


def get_remaining_lockout(ip):
    """Returns seconds remaining in lockout, or 0 if not locked."""
    now = time.time()
    with _lock:
        if ip in _attempts and len(_attempts[ip]) >= MAX_ATTEMPTS:
            oldest = _attempts[ip][0]
            remaining = LOCKOUT_SECONDS - (now - oldest)
            return max(0, int(remaining))
    return 0


def clear_attempts(ip):
    """Clear failed attempts for an IP after successful login."""
    with _lock:
        _attempts.pop(ip, None)


# ── MySQL credential validation ────────────────────────────────────────────


def validate_mysql_credentials(username, password, host=None, database=None):
    """Delegates to centralized data_layer."""
    return data_layer.validate_credentials(
        username, password, host=host, database=database
    )


# ── Django view decorator ───────────────────────────────────────────────────


def get_client_ip(request):
    """Extract client IP, handling X-Forwarded-For."""
    xff = request.META.get("HTTP_X_FORWARDED_FOR")
    if xff:
        # Take the first IP (client), ignore proxies
        return xff.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "127.0.0.1")


def login_required_mysql(view_func):
    """
    Decorator that requires the user to be authenticated via MySQL credentials.
    Redirects unauthenticated users to the login page.
    For AJAX/API endpoints, returns 401 JSON instead.
    """

    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.session.get("mysql_authenticated"):
            # Check if this is an API call (AJAX/JSON)
            if request.path.startswith("/api/"):
                return JsonResponse(
                    {"error": "Authentication required"},
                    status=401,
                )
            return redirect("predictor:login")
        return view_func(request, *args, **kwargs)

    return wrapper
