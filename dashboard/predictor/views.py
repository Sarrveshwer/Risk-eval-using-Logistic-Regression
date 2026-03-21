import json
import os
import datetime
import threading
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from . import ml_engine
from .auth import (
    login_required_mysql,
    validate_mysql_credentials,
    is_rate_limited,
    record_failed_attempt,
    clear_attempts,
    get_client_ip,
    get_remaining_lockout,
)


# ── Authentication Views ────────────────────────────────────────────────────


def login_view(request):
    """
    Login page — validates credentials by attempting a real MySQL connection.
    Protected against:
      - CSRF (Django middleware handles this via {% csrf_token %} in template)
      - Rate limiting (5 failed attempts per 5-minute window per IP)
      - Credential leakage (passwords never stored in session or logs)
    """
    # Already authenticated → go to dashboard
    if request.session.get("mysql_authenticated"):
        return redirect("predictor:dashboard")

    if request.method == "GET":
        return render(request, "predictor/login.html")

    # POST — authenticate
    ip = get_client_ip(request)

    # Check rate limit BEFORE processing credentials
    if is_rate_limited(ip):
        remaining = get_remaining_lockout(ip)
        return render(
            request,
            "predictor/login.html",
            {"error": f"Too many failed attempts. Try again in {remaining} seconds."},
        )

    username = request.POST.get("username", "").strip()
    password = request.POST.get("password", "")
    host = request.POST.get("host", "127.0.0.1").strip()
    database = request.POST.get("database", "ml_model").strip()

    # Validate input presence
    if not username or not password:
        return render(
            request,
            "predictor/login.html",
            {"error": "Username and password are required."},
        )

    # Attempt MySQL connection
    success, error_msg = validate_mysql_credentials(username, password, host=host, database=database)

    if success:
        # Clear rate-limit counter on success
        clear_attempts(ip)

        # Regenerate session ID to prevent session fixation attacks
        request.session.flush()
        request.session["mysql_authenticated"] = True
        request.session["mysql_user"] = username
        request.session["mysql_host"] = host
        request.session["mysql_db"] = database
        # Password is intentionally NOT stored in the session
        return redirect("predictor:dashboard")
    else:
        record_failed_attempt(ip)
        return render(
            request,
            "predictor/login.html",
            {"error": error_msg},
        )


def logout_view(request):
    """Logs out by flushing the entire session."""
    request.session.flush()
    return redirect("predictor:login")


# ── Protected Views ─────────────────────────────────────────────────────────


@login_required_mysql
def dashboard_view(request):
    """Main dashboard page."""
    if not request.session.session_key:
        request.session.create()
    session = ml_engine.get_session(
        request.session.session_key,
        host=request.session.get("mysql_host"),
        db=request.session.get("mysql_db"),
    )
    return render(
        request,
        "predictor/dashboard.html",
        {
            "step_log": session.step_log,
        },
    )


@login_required_mysql
def test_view(request):
    """Sensor input / test page."""
    if not request.session.session_key:
        request.session.create()
    return render(
        request,
        "predictor/test.html",
        {
            "sensor_cols": ml_engine.SENSOR_COLS,
            "presets": ml_engine.PRESET_NAMES,
        },
    )


@login_required_mysql
@csrf_exempt
def predict_api(request):
    """AJAX endpoint: receives sensor values, returns prediction."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    if not request.session.session_key:
        request.session.create()

    try:
        data = json.loads(request.body)
        sensor_values = {}
        for col in ml_engine.SENSOR_COLS:
            sensor_values[col] = float(data[col])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return JsonResponse({"error": f"Invalid input: {e}"}, status=400)

    session = ml_engine.get_session(
        request.session.session_key,
        host=request.session.get("mysql_host"),
        db=request.session.get("mysql_db"),
    )
    result = session.predict(sensor_values)
    return JsonResponse(result)


@login_required_mysql
@csrf_exempt
def reset_api(request):
    """AJAX endpoint: resets the prediction session."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    if not request.session.session_key:
        request.session.create()

    session = ml_engine.get_session(
        request.session.session_key,
        host=request.session.get("mysql_host"),
        db=request.session.get("mysql_db"),
    )
    session.preset_state["running"] = False  # Stop any background loops
    ml_engine.reset_session(
        request.session.session_key,
        host=request.session.get("mysql_host"),
        db=request.session.get("mysql_db"),
    )
    return JsonResponse({"status": "reset"})


@login_required_mysql
@csrf_exempt
def generate_preset_api(request):
    """AJAX endpoint: generates a 20-step preset scenario with random failure onset."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    try:
        data = json.loads(request.body)
        preset_name = data.get("preset", "heat_failure")
    except json.JSONDecodeError:
        preset_name = "heat_failure"

    if preset_name not in ml_engine.PRESET_NAMES:
        return JsonResponse({"error": f"Unknown preset: {preset_name}"}, status=400)

    result = ml_engine.generate_preset(
        preset_name,
        host=request.session.get("mysql_host"),
        database=request.session.get("mysql_db"),
    )
    return JsonResponse(result)


@login_required_mysql
@csrf_exempt
def run_preset_api(request):
    """
    AJAX endpoint: kicks off a preset scenario entirely on the server
    in a background daemon thread. Returns immediately. The client just
    keeps polling /api/log/ as normal — no JS loop needed.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    if not request.session.session_key:
        request.session.create()

    try:
        data = json.loads(request.body)
        preset_name = data.get("preset", "hdf")
    except json.JSONDecodeError:
        preset_name = "hdf"

    if preset_name not in ml_engine.PRESET_NAMES:
        return JsonResponse({"error": f"Unknown preset: {preset_name}"}, status=400)

    preset_data = ml_engine.generate_preset(
        preset_name,
        host=request.session.get("mysql_host"),
        database=request.session.get("mysql_db"),
    )
    rows = preset_data["rows"]
    failure_step = preset_data["failure_step"]

    # Get session. Only reset if it's a NEW preset or manual override.
    # If it's the 'database' loop restarting, we keep history for a continuous graph.
    session = ml_engine.get_session(
        request.session.session_key,
        host=request.session.get("mysql_host"),
        db=request.session.get("mysql_db"),
    )
    if preset_name != "database" or (not session.step_log):
        ml_engine.reset_session(
            request.session.session_key,
            host=request.session.get("mysql_host"),
            db=request.session.get("mysql_db"),
        )
        session = ml_engine.get_session(
            request.session.session_key,
            host=request.session.get("mysql_host"),
            db=request.session.get("mysql_db"),
        )
    else:
        # Just clear the 'done' state so the thread can start fresh
        session.preset_state["done"] = False
        session.preset_state["running"] = True

    thread = threading.Thread(
        target=ml_engine.run_preset_steps,
        args=(session, rows, failure_step),
        daemon=True,
    )
    thread.start()

    return JsonResponse(
        {
            "status": "started",
            "total_steps": len(rows),
            "failure_step": failure_step,
        }
    )


@login_required_mysql
def logs_view(request):
    """Logs page - list and view log files."""
    log_files = ml_engine.get_log_files()
    today_log = datetime.datetime.now().strftime("%Y-%m-%d.log")
    return render(
        request,
        "predictor/logs.html",
        {
            "log_files": log_files,
            "today_log": today_log,
        },
    )


@login_required_mysql
def read_log_ajax(request):
    """AJAX endpoint: returns a log file's content as JSON (GET)."""
    filename = request.GET.get("file", "")
    if not filename:
        return JsonResponse({"error": "No file specified"}, status=400)
    content = ml_engine.read_log_file(filename)
    return JsonResponse({"content": content, "filename": filename})


@login_required_mysql
def model_info_view(request):
    """Model info page - shows architecture and performance."""
    return render(
        request,
        "predictor/model_info.html",
        {
            "images_dir": ml_engine.IMAGES_DIR,
            "stats": ml_engine.get_model_stats(),
        },
    )


@login_required_mysql
def get_log(request):
    """AJAX endpoint: returns step log + preset state for current session."""
    if not request.session.session_key:
        request.session.create()
    session = ml_engine.get_session(
        request.session.session_key,
        host=request.session.get("mysql_host"),
        db=request.session.get("mysql_db"),
    )
    import time
    session.last_poll_time = time.time()
    return JsonResponse(
        {
            "log": session.step_log,
            "preset_state": session.preset_state,
            "first_critical_step": session.first_critical_step,
        }
    )
