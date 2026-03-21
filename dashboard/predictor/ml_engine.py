"""
ML Engine wrapper - loads the FailurePredictionSystem models
and holds per-session state for the Django dashboard.
"""

import os
import re
import sys
import time
import datetime
import threading
import pandas as pd
import joblib as jb
import random

try:
    from sqlalchemy import create_engine, text as _text

    _MYSQL_AVAILABLE = True
except ImportError:
    _MYSQL_AVAILABLE = False

# Add project root to sys.path so we can import data_layer
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
import data_layer

# Maps preset name → MySQL test table name
PRESET_TABLE_MAP = {
    "normal": "test_normal",
    "hdf": "test_hdf",
    "twf": "test_twf",
    "osf": "test_osf",
    "pwf": "test_pwf",
    "random_failure": "test_random_failure",
    "database": "smooth_simulation",
}

# Add the parent project directory to sys.path so we can import from main.py
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
FRONT_LOGS_DIR = os.path.join(PROJECT_ROOT, "log_front")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")

# Sensor columns expected by the model
SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET = "Machine failure"
CLASSIFICATIONS = ["TWF", "HDF", "PWF", "OSF"]
IGNORE_LIST = ["RNF", "UDI", "Product ID", "Type"] + CLASSIFICATIONS


# ── Test-run log helpers ────────────────────────────────────────────────────


def _test_log_path():
    """Returns the path for today's test_run log file from data_layer."""
    return data_layer.get_test_log_path()


def _write_test_log(line: str):
    """Appends a line via data_layer."""
    data_layer.write_test_log(line)


from model import FailurePredictionSystem, load_dataframe


class PredictionSession:
    """
    Holds state for one user session - wraps the FailurePredictionSystem
    from model.py to maintain rolling history and run predictions.
    """

    def __init__(self, mysql_host=None, mysql_db=None):
        self.mysql_host = mysql_host
        self.mysql_db = mysql_db
        # We load a small df just to initialize the system if models need training,
        # but usually they are already on disk.
        # In a real production app, we might want to avoid loading the whole training DF here.
        # However, for this project, FailurePredictionSystem expects it.
        # To optimize, we can pass an empty DF if we know models exist.

        # Load a dummy or minimal DF if models exist to save memory
        try:
            df = pd.DataFrame(
                columns=SENSOR_COLS + [TARGET] + CLASSIFICATIONS + IGNORE_LIST
            )
        except:
            df = pd.DataFrame()

        self.system = FailurePredictionSystem(
            df=df,
            target=TARGET,
            risk_tolerance=0.50,  # Lowered from 0.70/0.765 to improve recall
            ignore=IGNORE_LIST,
            classifications=CLASSIFICATIONS,
            warning_sensitivity=0.6,
            diagnosis_sensitivity=0.3,
            persistence_threshold=2,
        )
        self.system.LoadModels(model_dir=MODELS_DIR)

        self.step_log = []  # stores recent prediction results (FIFO)
        self.total_steps_seen = 0
        self.first_critical_step = None
        self.preset_state = {
            "running": False,
            "total": 0,
            "failure_step": None,
            "done": False,
        }
        self.last_poll_time = time.time()

    def reset(self):
        self.system.history = []
        if hasattr(self.system, "prob_history"):
            self.system.prob_history = []
        self.system.warning_streak = 0
        self.step_log = []
        self.total_steps_seen = 0
        self.first_critical_step = None
        self.preset_state = {
            "running": False,
            "total": 0,
            "failure_step": None,
            "done": False,
        }
        self.last_poll_time = time.time()
        _write_test_log("--- Session Reset ---")

    def predict(self, sensor_values):
        """
        Takes a dict of sensor values, runs through the FailurePredictionSystem,
        returns prediction result dict.
        """
        # Build a DataFrame row
        row = {col: float(sensor_values[col]) for col in SENSOR_COLS}
        # Add placeholders for metadata columns expected by the system logic
        row[TARGET] = 0
        for c in CLASSIFICATIONS:
            row[c] = 0
        for ig in ["RNF", "UDI", "Product ID", "Type"]:
            row[ig] = 0

        X = pd.DataFrame([row])

        # Use the logic from FailurePredictionSystem
        sys_result = self.system.predict(X)

        self.total_steps_seen += 1
        step_num = self.total_steps_seen

        # Format the result to match what the dashboard expects
        result = {
            "step": step_num,
            "risk_prob": sys_result["risk_prob"],
            "alert_level": sys_result["alert_level"],
            "failure_type": (
                sys_result["failure_type"]
                if sys_result["alert_level"] != "HEALTHY"
                else None
            ),
            "streak": self.system.warning_streak,
            "details": sys_result["details"],
            "sensors": {col: float(sensor_values[col]) for col in SENSOR_COLS},
        }

        self.step_log.append(result)
        if len(self.step_log) > 50:
            self.step_log.pop(0)

        # Track first CRITICAL prediction for verdict display
        if sys_result["alert_level"] == "CRITICAL" and self.first_critical_step is None:
            self.first_critical_step = step_num

        # ── Write to test-run log ──
        s = sensor_values
        _write_test_log(
            f"Step {step_num:<3} | Risk: {result['risk_prob']:.4f} | Alert: {result['alert_level']:<8} "
            f"| Failure: {str(result['failure_type']):<15} "
            f"| Air: {float(s['Air temperature [K]']):.1f} "
            f"| ProcT: {float(s['Process temperature [K]']):.1f} "
            f"| RPM: {int(s['Rotational speed [rpm]'])} "
            f"| Torque: {float(s['Torque [Nm]']):.1f} "
            f"| Wear: {int(s['Tool wear [min]'])}"
        )

        return result


# ── Dynamic preset generator ────────────────────────────────────────────────

# Normal operating ranges for each sensor
NORMAL = {
    "air": (298, 302),
    "proc": (308, 312),
    "rpm": (1450, 1550),
    "torque": (38, 42),
    "wear": (50, 120),
}

# Failure extremes — what sensor values look like when the machine actually fails
FAILURE_EXTREMES = {
    "heat_failure": {"air": 315, "proc": 325, "rpm": 900, "torque": 70, "wear": 130},
    "tool_wear": {"air": 302, "proc": 312, "rpm": 1500, "torque": 55, "wear": 240},
    "overstrain": {"air": 303, "proc": 313, "rpm": 800, "torque": 80, "wear": 130},
    "power_failure": {"air": 305, "proc": 315, "rpm": 2500, "torque": 60, "wear": 140},
}

PRESET_NAMES = ["normal", "hdf", "twf", "osf", "pwf", "random_failure", "database"]

# Ambiguous extremes for random failure — slight degradation across ALL sensors,
# not enough to confidently match any single known failure mode.
RANDOM_FAILURE_EXTREMES = {
    "air": 308,
    "proc": 318,
    "rpm": 1200,
    "torque": 58,
    "wear": 180,
}


def _fetch_db_rows(preset_name, host=None, database=None):
    """Delegates to data_layer."""
    try:
        return data_layer.fetch_simulation_rows(preset_name, host=host, database=database)
    except Exception as e:
        print(f"[_fetch_db_rows] Connection lost or error: {e}")
        return None


def generate_preset(preset_name, host=None, database=None):
    """
    Generate 20 rows of sensor data for a preset scenario.

    Priority:
      1. Real rows from the MySQL test table (fetched randomly).
      2. Synthetic fallback when MySQL is unreachable or the table is empty.

    For failure presets the last row is always replaced with the
    failure-extreme values so the 'MACHINE FAILED' overlay fires correctly.

    Returns: {'rows': [[air, proc, rpm, torque, wear], ...], 'failure_step': int|None}
    """
    rows = []

    # ── Attempt MySQL first (Only for continuous simulation or normal data) ───
    # We skip MySQL fetch for specific failure presets (hdf, twf, etc.) because 
    # random rows from the DB break the time-series trend features the model needs.
    # The synthetic ramp-up logic below is better for test cases.
    if preset_name in ["normal", "database"]:
        db_rows = _fetch_db_rows(preset_name, host=host, database=database)
    else:
        db_rows = None

    if db_rows:
        if preset_name in ["normal", "database"]:
            return {"rows": db_rows, "failure_step": None}
        # For all failure presets: cap at 19 normal + 1 extreme final row
        db_rows = db_rows[:19]
        extremes_key = {
            "hdf": "heat_failure",
            "twf": "tool_wear",
            "osf": "overstrain",
            "pwf": "power_failure",
            "random_failure": None,  # uses RANDOM_FAILURE_EXTREMES below
        }.get(preset_name)
        if extremes_key and extremes_key in FAILURE_EXTREMES:
            ex = FAILURE_EXTREMES[extremes_key]
            db_rows.append([ex["air"], ex["proc"], ex["rpm"], ex["torque"], ex["wear"]])
        elif extremes_key == "power_failure":
            # Just fallback to random for pwf as no extreme is defined yet
            ex = RANDOM_FAILURE_EXTREMES
            db_rows.append([ex["air"], ex["proc"], ex["rpm"], ex["torque"], ex["wear"]])
        else:
            ex = RANDOM_FAILURE_EXTREMES
            db_rows.append([ex["air"], ex["proc"], ex["rpm"], ex["torque"], ex["wear"]])
        return {"rows": db_rows, "failure_step": len(db_rows)}

    # ── Synthetic fallback (original logic) ──────────────────────────────────
    if preset_name in ["normal", "database"]:
        # 20 steps of normal operation with small jitter
        for i in range(20):
            rows.append(
                [
                    round(random.uniform(*NORMAL["air"]), 1),
                    round(random.uniform(*NORMAL["proc"]), 1),
                    round(random.uniform(*NORMAL["rpm"])),
                    round(random.uniform(*NORMAL["torque"]), 1),
                    round(NORMAL["wear"][0] + i * 2),
                ]
            )
        return {"rows": rows, "failure_step": None}

    if preset_name == "random_failure":
        # Random failure: ambiguous degradation across ALL sensors simultaneously.
        # No single failure mode dominates, so the classifier can't reach diagnosis_sensitivity
        # on any of TWF/HDF/PWF/OSF — the runtime correctly falls back to "RandomFailure".
        onset = random.randint(6, 13)
        extremes = RANDOM_FAILURE_EXTREMES
        ramp_steps = 20 - onset - 1

        for i in range(20):
            if i < onset:
                rows.append(
                    [
                        round(random.uniform(*NORMAL["air"]), 1),
                        round(random.uniform(*NORMAL["proc"]), 1),
                        round(random.uniform(*NORMAL["rpm"])),
                        round(random.uniform(*NORMAL["torque"]), 1),
                        round(NORMAL["wear"][0] + i * 2),
                    ]
                )
            elif i < 19:
                progress = (i - onset) / max(ramp_steps, 1)
                noise = lambda: random.uniform(-0.5, 0.5)
                air_n = random.uniform(*NORMAL["air"])
                proc_n = random.uniform(*NORMAL["proc"])
                rpm_n = random.uniform(*NORMAL["rpm"])
                torq_n = random.uniform(*NORMAL["torque"])
                wear_n = NORMAL["wear"][0] + i * 2
                rows.append(
                    [
                        round(
                            air_n + (extremes["air"] - air_n) * progress + noise(), 1
                        ),
                        round(
                            proc_n + (extremes["proc"] - proc_n) * progress + noise(), 1
                        ),
                        round(rpm_n + (extremes["rpm"] - rpm_n) * progress),
                        round(
                            torq_n + (extremes["torque"] - torq_n) * progress + noise(),
                            1,
                        ),
                        round(wear_n + (extremes["wear"] - wear_n) * progress),
                    ]
                )
            else:
                rows.append(
                    [
                        extremes["air"],
                        extremes["proc"],
                        extremes["rpm"],
                        extremes["torque"],
                        extremes["wear"],
                    ]
                )
        return {"rows": rows, "failure_step": 20 if preset_name != "database" else None}

    # For the named failure presets: random onset between step 6 and 13
    onset = random.randint(6, 13)

    extremes_key = {
        "hdf": "heat_failure",
        "twf": "tool_wear",
        "osf": "overstrain",
        "pwf": "power_failure",
    }.get(preset_name, "heat_failure")

    extremes = FAILURE_EXTREMES.get(extremes_key, FAILURE_EXTREMES["heat_failure"])
    ramp_steps = 20 - onset - 1  # steps from onset to step 19 (step 20 is the failure)

    for i in range(20):
        if i < onset:
            # Normal operation with slight jitter
            rows.append(
                [
                    round(random.uniform(*NORMAL["air"]), 1),
                    round(random.uniform(*NORMAL["proc"]), 1),
                    round(random.uniform(*NORMAL["rpm"])),
                    round(random.uniform(*NORMAL["torque"]), 1),
                    round(NORMAL["wear"][0] + i * 2),
                ]
            )
        elif i < 19:
            # Gradually ramp toward failure
            progress = (i - onset) / max(ramp_steps, 1)
            air_norm = random.uniform(*NORMAL["air"])
            proc_norm = random.uniform(*NORMAL["proc"])
            rpm_norm = random.uniform(*NORMAL["rpm"])
            torque_norm = random.uniform(*NORMAL["torque"])
            wear_norm = NORMAL["wear"][0] + i * 2

            rows.append(
                [
                    round(air_norm + (extremes["air"] - air_norm) * progress, 1),
                    round(proc_norm + (extremes["proc"] - proc_norm) * progress, 1),
                    round(rpm_norm + (extremes["rpm"] - rpm_norm) * progress),
                    round(
                        torque_norm + (extremes["torque"] - torque_norm) * progress, 1
                    ),
                    round(wear_norm + (extremes["wear"] - wear_norm) * progress),
                ]
            )
        else:
            # Step 20: Machine actually fails — extreme values
            rows.append(
                [
                    extremes["air"],
                    extremes["proc"],
                    extremes["rpm"],
                    extremes["torque"],
                    extremes["wear"],
                ]
            )

    return {"rows": rows, "failure_step": 20 if preset_name != "database" else None}


# ── In-memory session store (keyed by Django session ID) ───────────────────
_sessions = {}


def get_session(session_id, host=None, db=None):
    if session_id not in _sessions:
        _sessions[session_id] = PredictionSession(mysql_host=host, mysql_db=db)
    return _sessions[session_id]


def reset_session(session_id, host=None, db=None):
    if session_id in _sessions:
        _sessions[session_id].reset()
    else:
        _sessions[session_id] = PredictionSession(mysql_host=host, mysql_db=db)


def run_preset_steps(session, rows, failure_step):
    """
    Run preset steps server-side, one per second.
    Called in a daemon background thread — keeps running regardless of
    whether the browser is on the dashboard, logs page, or anywhere else.
    """
    SENSOR_KEYS = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    session.preset_state = {
        "running": True,
        "total": len(rows),
        "failure_step": failure_step,
        "done": False,
    }

    for i, row_values in enumerate(rows):
        # Check kill-switch
        if not session.preset_state.get("running"):
            print(f"[run_preset_steps] Thread received stop signal at step {i + 1}")
            break

        # Check for client disconnect (no polling for > 8 seconds)
        if hasattr(session, "last_poll_time") and time.time() - session.last_poll_time > 8.0:
            print(f"[run_preset_steps] Client disconnected (stale poll). Aborting test case and clearing cache.")
            session.preset_state["running"] = False
            session.reset()
            break

        # 1-2 sec interval
        time.sleep(random.uniform(1.0, 2.0))
        sensor_values = {k: v for k, v in zip(SENSOR_KEYS, row_values)}
        try:
            session.predict(sensor_values)
        except Exception as e:
            print(f"[run_preset_steps] Error at step {i + 1}: {e}")
            break

    session.preset_state["running"] = False
    session.preset_state["done"] = True


def get_log_files():
    """Delegates to data_layer."""
    return data_layer.get_log_files()


def read_log_file(filename):
    """Delegates to data_layer."""
    return data_layer.read_log_file(filename)


# ── Dynamic model stats (parsed from most recent training log) ──────────────


def get_model_stats():
    """
    Returns a dict of model performance metrics parsed from the most recent
    main.py or model.py training log, plus live values from the loaded model.
    Falls back to None for any value that can't be parsed.
    """
    stats = {
        "roc_auc": None,
        "risk_tolerance": 0.765,
        "recall": None,
        "precision": None,
        "fpr": None,
        "tp": None,
        "fp": None,
        "fn": None,
        "tn": None,
        "feature_count": None,
        "log_file_used": None,
        "secondary_metrics": None,
    }

    if not os.path.exists(LOGS_DIR):
        return stats

    # Collect all main.py and model.py log files, sorted most-recent first
    candidate_logs = sorted(
        [
            f
            for f in os.listdir(LOGS_DIR)
            if f.endswith(".log")
            and (f.startswith("main.py@") or f.startswith("model.py@"))
        ],
        reverse=True,
    )

    # Walk from most recent and pick the first one that contains training metrics
    for logfile in candidate_logs:
        path = os.path.join(LOGS_DIR, logfile)
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError:
            continue

        # Only consider logs that have model evaluation output
        if "ROC-AUC score" not in content:
            continue

        stats["log_file_used"] = logfile

        # ROC-AUC
        m = re.search(r"ROC-AUC score\s*:\s*([0-9.]+)", content)
        if m:
            stats["roc_auc"] = float(m.group(1))

        # Risk tolerance
        m = re.search(r"Risk Tolerance used\s*:\s*([0-9.]+)", content)
        if m:
            stats["risk_tolerance"] = float(m.group(1))

        # TP / FP
        m = re.search(r"TP:\s*(\d+)\s*\|\s*FP:\s*(\d+)", content)
        if m:
            stats["tp"] = int(m.group(1))
            stats["fp"] = int(m.group(2))

        # FN / TN
        m = re.search(r"FN:\s*(\d+)\s*\|\s*TN:\s*(\d+)", content)
        if m:
            stats["fn"] = int(m.group(1))
            stats["tn"] = int(m.group(2))

        # Recall
        m = re.search(r"Recall \(TPR\)\s*:\s*([0-9.]+)", content)
        if m:
            stats["recall"] = float(m.group(1))

        # Precision
        m = re.search(r"Precision\s*:\s*([0-9.]+)", content)
        if m:
            stats["precision"] = float(m.group(1))

        # FPR
        m = re.search(r"False PosRate\s*:\s*([0-9.]+)", content)
        if m:
            stats["fpr"] = float(m.group(1))

        # Secondary model metrics (Classification Report)
        sec_metrics = {}
        # Parse per class 0-3 (TWF, HDF, PWF, OSF)
        class_names = ["TWF", "HDF", "PWF", "OSF"]
        for i, class_name in enumerate(class_names):
            # matches e.g.: " 0       0.53      0.89      0.67         9"
            pattern = rf"\s+{i}\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)"
            m2 = re.search(pattern, content)
            if m2:
                sec_metrics[class_name] = {
                    "precision": float(m2.group(1)),
                    "recall": float(m2.group(2)),
                    "f1": float(m2.group(3)),
                    "support": int(m2.group(4)),
                }

        # Parse accuracy and macro avg
        m_acc = re.search(r"accuracy\s+([0-9.]+)\s+([0-9]+)", content)
        if m_acc:
            sec_metrics["accuracy"] = float(m_acc.group(1))
            sec_metrics["total_support"] = int(m_acc.group(2))

        m_macro = re.search(
            r"macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)", content
        )
        if m_macro:
            sec_metrics["macro_precision"] = float(m_macro.group(1))
            sec_metrics["macro_recall"] = float(m_macro.group(2))
            sec_metrics["macro_f1"] = float(m_macro.group(3))

        if sec_metrics:
            stats["secondary_metrics"] = sec_metrics

        break  # found a usable log — stop searching

    # Live feature count from model
    try:
        tmp_session = list(_sessions.values())[0] if _sessions else PredictionSession()
        stats["feature_count"] = len(tmp_session.system.risk_model.feature_names_in_)
    except Exception:
        pass

    return stats
