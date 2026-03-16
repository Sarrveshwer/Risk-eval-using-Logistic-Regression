import json
import os
import threading
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from . import ml_engine


def dashboard_view(request):
    """Main dashboard page."""
    if not request.session.session_key:
        request.session.create()
    session = ml_engine.get_session(request.session.session_key)
    return render(
        request,
        "predictor/dashboard.html",
        {
            "step_log": session.step_log,
        },
    )


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

    session = ml_engine.get_session(request.session.session_key)
    result = session.predict(sensor_values)
    return JsonResponse(result)


@csrf_exempt
def reset_api(request):
    """AJAX endpoint: resets the prediction session."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=400)

    if not request.session.session_key:
        request.session.create()

    ml_engine.reset_session(request.session.session_key)
    return JsonResponse({"status": "reset"})


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

    result = ml_engine.generate_preset(preset_name)
    return JsonResponse(result)


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
        preset_name = data.get("preset", "heat_failure")
    except json.JSONDecodeError:
        preset_name = "heat_failure"

    if preset_name not in ml_engine.PRESET_NAMES:
        return JsonResponse({"error": f"Unknown preset: {preset_name}"}, status=400)

    preset_data = ml_engine.generate_preset(preset_name)
    rows = preset_data["rows"]
    failure_step = preset_data["failure_step"]

    # Reset session first, then get a fresh one
    ml_engine.reset_session(request.session.session_key)
    session = ml_engine.get_session(request.session.session_key)

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


def logs_view(request):
    """Logs page - list and view log files."""
    log_files = ml_engine.get_log_files()
    return render(
        request,
        "predictor/logs.html",
        {
            "log_files": log_files,
        },
    )


def read_log_ajax(request):
    """AJAX endpoint: returns a log file's content as JSON (GET)."""
    filename = request.GET.get("file", "")
    if not filename:
        return JsonResponse({"error": "No file specified"}, status=400)
    content = ml_engine.read_log_file(filename)
    return JsonResponse({"content": content, "filename": filename})


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


def get_log(request):
    """AJAX endpoint: returns step log + preset state for current session."""
    if not request.session.session_key:
        request.session.create()
    session = ml_engine.get_session(request.session.session_key)
    return JsonResponse(
        {
            "log": session.step_log,
            "preset_state": session.preset_state,
            "first_critical_step": session.first_critical_step,
        }
    )
