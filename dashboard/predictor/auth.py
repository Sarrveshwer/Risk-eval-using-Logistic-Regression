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

# ── Rate limiter (per-IP, in-memory, thread-safe) ──────────────────────────

_lock = threading.Lock()
_attempts = {}  # ip -> [timestamp, ...]

MAX_ATTEMPTS = 5         # max failed attempts per window
WINDOW_SECONDS = 300     # 5-minute window
LOCKOUT_SECONDS = 300    # 5-minute lockout after exceeding limit


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

def validate_mysql_credentials(username, password, host="127.0.0.1", port=3306):
    """
    Attempts a real MySQL connection with the provided credentials.
    Returns (True, None) on success, (False, error_message) on failure.

    Security notes:
      - Uses pymysql directly for minimal overhead
      - Connection is immediately closed after validation
      - Error messages are generic to avoid leaking server internals
    """
    if not username or not password:
        return False, "Username and password are required."

    # Sanity checks on input length/characters
    if len(username) > 80 or len(password) > 128:
        return False, "Invalid credentials."

    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=username,
            password=password,
            connect_timeout=5,
        )
        conn.close()
        return True, None
    except pymysql.err.OperationalError:
        # Covers: access denied, host not allowed, too many connections, etc.
        return False, "Authentication failed. Check your MySQL credentials."
    except pymysql.err.InterfaceError:
        return False, "Cannot reach the database server."
    except Exception:
        # Catch-all: never expose internal error details
        return False, "Authentication failed."


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
