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

try:
    from sqlalchemy import create_engine, text as _text
    _MYSQL_AVAILABLE = True
except ImportError:
    _MYSQL_AVAILABLE = False

# ── MySQL connection config (same credentials used for training data) ─────────
MYSQL_CONFIG = {
    'host':     'localhost',
    'user':     'root',
    'password': 'root',
    'database': 'ml_model',
}

# Maps preset name → MySQL test table name
PRESET_TABLE_MAP = {
    'normal':       'test_normal',
    'heat_failure': 'test_hdf',
    'tool_wear':    'test_twf',
    'overstrain':   'test_osf',
    'random_failure': 'test_rnf',
}

# Add the parent project directory to sys.path so we can import from main.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')

# Sensor columns expected by the model
SENSOR_COLS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
]

TARGET = 'Machine failure'
CLASSIFICATIONS = ['TWF', 'HDF', 'PWF', 'OSF']
IGNORE_LIST = ['RNF', 'UDI', 'Product ID', 'Type'] + CLASSIFICATIONS


# ── Test-run log helpers ────────────────────────────────────────────────────

def _test_log_path():
    """Returns the path for today's test_run log file."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    return os.path.join(LOGS_DIR, f'test_run_{date_str}.log')


def _write_test_log(line: str):
    """Appends a line (with timestamp) to today's test_run log and flushes immediately."""
    path = _test_log_path()
    timestamp = datetime.datetime.now().strftime('[%H:%M:%S] ')
    with open(path, 'a', encoding='utf-8') as f:
        f.write(timestamp + line + '\n')


class PredictionSession:
    """
    Holds state for one user session - wraps the loaded sklearn models
    and maintains the rolling history needed by predict().
    """

    def __init__(self):
        self.risk_model = jb.load(os.path.join(MODELS_DIR, 'Risk_eval_model.pkl'))
        self.classification_model = jb.load(os.path.join(MODELS_DIR, 'Risk_eval_Classification.pkl'))
        self.history = []
        self.prob_history = []
        self.warning_streak = 0
        self.risk_tolerance = 0.765
        self.warning_sensitivity = 0.6
        self.diagnosis_sensitivity = 0.4
        self.persistence_threshold = 3
        self.step_log = []  # stores all prediction results
        self.first_critical_step = None
        self.preset_state = {'running': False, 'total': 0, 'failure_step': None, 'done': False}

    def reset(self):
        self.history = []
        self.prob_history = []
        self.warning_streak = 0
        self.step_log = []
        self.first_critical_step = None
        self.preset_state = {'running': False, 'total': 0, 'failure_step': None, 'done': False}
        _write_test_log('--- Session Reset ---')

    def predict(self, sensor_values):
        """
        Takes a dict of sensor values, runs through the two-tier model,
        returns prediction result dict.
        sensor_values: dict with keys matching SENSOR_COLS
        """
        # Build a DataFrame row
        row = {col: float(sensor_values[col]) for col in SENSOR_COLS}
        # Add placeholder columns for target and classifications (needed by feature builder)
        row[TARGET] = 0
        for c in CLASSIFICATIONS:
            row[c] = 0
        for ig in ['RNF', 'UDI', 'Product ID', 'Type']:
            row[ig] = 0

        X = pd.DataFrame([row])

        # Store in history
        current_row_dict = X.iloc[0].to_dict()
        self.history.append(current_row_dict)
        if len(self.history) > 5:
            self.history.pop(0)

        current_data_df = X.copy()
        history_dataframe = pd.DataFrame(self.history)

        cols_to_exclude = [TARGET] + CLASSIFICATIONS + IGNORE_LIST

        dynamic_features = [
            c for c in X.columns
            if c not in cols_to_exclude
            and not any(x in c for x in ['_Rolling_Mean', '_Volatility', '_Delta', '_Rolling_Delta'])
        ]

        for col in dynamic_features:
            current_data_df[f'{col}_Rolling_Mean'] = history_dataframe[col].mean()

            if len(self.history) > 1:
                current_data_df[f'{col}_Volatility'] = history_dataframe[col].std()
                current_data_df[f'{col}_Delta'] = self.history[-1][col] - self.history[-2][col]
                current_data_df[f'{col}_Rolling_Delta'] = history_dataframe[col].diff().mean()
            else:
                current_data_df[f'{col}_Volatility'] = 0.0
                current_data_df[f'{col}_Delta'] = 0.0
                current_data_df[f'{col}_Rolling_Delta'] = 0.0

        current_data_df = current_data_df.fillna(0)

        inference_df = current_data_df.drop(columns=[c for c in cols_to_exclude if c in current_data_df.columns])
        inference_df = inference_df[self.risk_model.feature_names_in_]

        # Tier 1: Risk Probability
        raw_risk_prob = float(self.risk_model.predict_proba(inference_df)[:, 1][0])

        self.prob_history.append(raw_risk_prob)
        if len(self.prob_history) > 3:
            self.prob_history.pop(0)

        risk_prob = sum(self.prob_history) / len(self.prob_history)

        warning_zone = self.risk_tolerance * self.warning_sensitivity

        if risk_prob >= warning_zone:
            self.warning_streak += 1
        else:
            self.warning_streak = 0

        if risk_prob >= self.risk_tolerance:
            alert_status = "CRITICAL"
        elif risk_prob >= 0.60:
            alert_status = "WARNING"
            self.warning_streak = max(self.warning_streak, self.persistence_threshold)
        elif risk_prob >= warning_zone:
            if self.warning_streak >= self.persistence_threshold:
                alert_status = "WARNING"
            else:
                alert_status = "HEALTHY"
        else:
            alert_status = "HEALTHY"

        # Tier 2: Softmax Diagnosis
        softmax_probs = self.classification_model.predict_proba(inference_df)[0]
        model_classes = self.classification_model.classes_
        results = {label: 0.0 for label in CLASSIFICATIONS}

        # model_classes contains integer class indices (0..3); each maps to CLASSIFICATIONS[idx]
        for i, prob in enumerate(softmax_probs):
            class_idx = int(model_classes[i])
            if class_idx < len(CLASSIFICATIONS):
                label = CLASSIFICATIONS[class_idx]
                results[label] = round(float(prob), 4)

        active = [f for f, p in results.items() if p >= self.diagnosis_sensitivity]
        # When none of the 4 known failure types are confident enough, fall back to RandomFailure
        failure = max(results, key=results.get) if active else "RandomFailure"

        step_num = len(self.step_log) + 1
        result = {
            "step": step_num,
            "risk_prob": round(risk_prob, 4),
            "alert_level": alert_status,
            "failure_type": failure if alert_status != "HEALTHY" else None,
            "streak": self.warning_streak,
            "details": results,
            "sensors": {col: float(sensor_values[col]) for col in SENSOR_COLS},
        }

        self.step_log.append(result)

        # Track first CRITICAL prediction for verdict display
        if alert_status == 'CRITICAL' and self.first_critical_step is None:
            self.first_critical_step = step_num

        # ── Write to test-run log (real-time, flushed immediately) ──
        s = sensor_values
        _write_test_log(
            f"Step {step_num:<3} | Risk: {risk_prob:.4f} | Alert: {alert_status:<8} "
            f"| Failure: {str(result['failure_type']):<15} "
            f"| Air: {float(s['Air temperature [K]']):.1f} "
            f"| ProcT: {float(s['Process temperature [K]']):.1f} "
            f"| RPM: {int(s['Rotational speed [rpm]'])} "
            f"| Torque: {float(s['Torque [Nm]']):.1f} "
            f"| Wear: {int(s['Tool wear [min]'])}"
        )

        return result


# ── Dynamic preset generator ────────────────────────────────────────────────
import random


# Normal operating ranges for each sensor
NORMAL = {
    'air': (298, 302),
    'proc': (308, 312),
    'rpm': (1450, 1550),
    'torque': (38, 42),
    'wear': (50, 120),
}

# Failure extremes — what sensor values look like when the machine actually fails
FAILURE_EXTREMES = {
    'heat_failure': {'air': 315, 'proc': 325, 'rpm': 900, 'torque': 70, 'wear': 130},
    'tool_wear':    {'air': 302, 'proc': 312, 'rpm': 1500, 'torque': 55, 'wear': 240},
    'overstrain':   {'air': 303, 'proc': 313, 'rpm': 800, 'torque': 80, 'wear': 130},
}

PRESET_NAMES = ['normal', 'heat_failure', 'tool_wear', 'overstrain', 'random_failure']

# Ambiguous extremes for random failure — slight degradation across ALL sensors,
# not enough to confidently match any single known failure mode.
RANDOM_FAILURE_EXTREMES = {
    'air': 308, 'proc': 318, 'rpm': 1200, 'torque': 58, 'wear': 180
}


def _fetch_db_rows(preset_name):
    """
    Try to load 20 sensor rows from the corresponding MySQL test table.
    Returns a list of [air, proc, rpm, torque, wear] rows, or None on failure.
    """
    if not _MYSQL_AVAILABLE:
        return None
    table = PRESET_TABLE_MAP.get(preset_name)
    if not table:
        return None
    
    url = (
        f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}"
        f"@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}"
    )
    
    try:
        engine = create_engine(url)
        with engine.connect() as conn:
            result = conn.execute(_text(
                f"SELECT `Air temperature [K]`, `Process temperature [K]`, "
                f"`Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]` "
                f"FROM `{table}` ORDER BY RAND() LIMIT 20"
            ))
            rows = result.fetchall()
            if not rows:
                return None
            return [list(r) for r in rows]
    except Exception as e:
        print(f"[_fetch_db_rows] MySQL Error: {e}")
        return None


def generate_preset(preset_name):
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

    # ── Attempt MySQL first ───────────────────────────────────────────────────
    db_rows = _fetch_db_rows(preset_name)
    if db_rows:
        if preset_name == 'normal':
            return {'rows': db_rows, 'failure_step': None}
        # For all failure presets: cap at 19 normal + 1 extreme final row
        db_rows = db_rows[:19]
        extremes_key = {
            'heat_failure': 'heat_failure',
            'tool_wear':    'tool_wear',
            'overstrain':   'overstrain',
            'random_failure': None,  # uses RANDOM_FAILURE_EXTREMES below
        }.get(preset_name)
        if extremes_key:
            ex = FAILURE_EXTREMES[extremes_key]
            db_rows.append([ex['air'], ex['proc'], ex['rpm'], ex['torque'], ex['wear']])
        else:
            ex = RANDOM_FAILURE_EXTREMES
            db_rows.append([ex['air'], ex['proc'], ex['rpm'], ex['torque'], ex['wear']])
        return {'rows': db_rows, 'failure_step': len(db_rows)}

    # ── Synthetic fallback (original logic) ──────────────────────────────────
    if preset_name == 'normal':
        # 20 steps of normal operation with small jitter
        for i in range(20):
            rows.append([
                round(random.uniform(*NORMAL['air']), 1),
                round(random.uniform(*NORMAL['proc']), 1),
                round(random.uniform(*NORMAL['rpm'])),
                round(random.uniform(*NORMAL['torque']), 1),
                round(NORMAL['wear'][0] + i * 2),
            ])
        return {'rows': rows, 'failure_step': None}

    if preset_name == 'random_failure':
        # Random failure: ambiguous degradation across ALL sensors simultaneously.
        # No single failure mode dominates, so the classifier can't reach diagnosis_sensitivity
        # on any of TWF/HDF/PWF/OSF — the runtime correctly falls back to "RandomFailure".
        onset = random.randint(6, 13)
        extremes = RANDOM_FAILURE_EXTREMES
        ramp_steps = 20 - onset - 1

        for i in range(20):
            if i < onset:
                rows.append([
                    round(random.uniform(*NORMAL['air']), 1),
                    round(random.uniform(*NORMAL['proc']), 1),
                    round(random.uniform(*NORMAL['rpm'])),
                    round(random.uniform(*NORMAL['torque']), 1),
                    round(NORMAL['wear'][0] + i * 2),
                ])
            elif i < 19:
                progress = (i - onset) / max(ramp_steps, 1)
                noise = lambda: random.uniform(-0.5, 0.5)
                air_n  = random.uniform(*NORMAL['air'])
                proc_n = random.uniform(*NORMAL['proc'])
                rpm_n  = random.uniform(*NORMAL['rpm'])
                torq_n = random.uniform(*NORMAL['torque'])
                wear_n = NORMAL['wear'][0] + i * 2
                rows.append([
                    round(air_n  + (extremes['air']    - air_n)  * progress + noise(), 1),
                    round(proc_n + (extremes['proc']   - proc_n) * progress + noise(), 1),
                    round(rpm_n  + (extremes['rpm']    - rpm_n)  * progress),
                    round(torq_n + (extremes['torque'] - torq_n) * progress + noise(), 1),
                    round(wear_n + (extremes['wear']   - wear_n) * progress),
                ])
            else:
                rows.append([
                    extremes['air'], extremes['proc'], extremes['rpm'],
                    extremes['torque'], extremes['wear'],
                ])
        return {'rows': rows, 'failure_step': 20}

    # For the named failure presets: random onset between step 6 and 13
    onset = random.randint(6, 13)
    extremes = FAILURE_EXTREMES[preset_name]
    ramp_steps = 20 - onset - 1  # steps from onset to step 19 (step 20 is the failure)

    for i in range(20):
        if i < onset:
            # Normal operation with slight jitter
            rows.append([
                round(random.uniform(*NORMAL['air']), 1),
                round(random.uniform(*NORMAL['proc']), 1),
                round(random.uniform(*NORMAL['rpm'])),
                round(random.uniform(*NORMAL['torque']), 1),
                round(NORMAL['wear'][0] + i * 2),
            ])
        elif i < 19:
            # Gradually ramp toward failure
            progress = (i - onset) / max(ramp_steps, 1)
            air_norm = random.uniform(*NORMAL['air'])
            proc_norm = random.uniform(*NORMAL['proc'])
            rpm_norm = random.uniform(*NORMAL['rpm'])
            torque_norm = random.uniform(*NORMAL['torque'])
            wear_norm = NORMAL['wear'][0] + i * 2

            rows.append([
                round(air_norm + (extremes['air'] - air_norm) * progress, 1),
                round(proc_norm + (extremes['proc'] - proc_norm) * progress, 1),
                round(rpm_norm + (extremes['rpm'] - rpm_norm) * progress),
                round(torque_norm + (extremes['torque'] - torque_norm) * progress, 1),
                round(wear_norm + (extremes['wear'] - wear_norm) * progress),
            ])
        else:
            # Step 20: Machine actually fails — extreme values
            rows.append([
                extremes['air'],
                extremes['proc'],
                extremes['rpm'],
                extremes['torque'],
                extremes['wear'],
            ])

    return {'rows': rows, 'failure_step': 20}


# ── In-memory session store (keyed by Django session ID) ───────────────────
_sessions = {}


def get_session(session_id):
    if session_id not in _sessions:
        _sessions[session_id] = PredictionSession()
    return _sessions[session_id]


def reset_session(session_id):
    if session_id in _sessions:
        _sessions[session_id].reset()
    else:
        _sessions[session_id] = PredictionSession()


def run_preset_steps(session, rows, failure_step):
    """
    Run preset steps server-side, one per second.
    Called in a daemon background thread — keeps running regardless of
    whether the browser is on the dashboard, logs page, or anywhere else.
    """
    SENSOR_KEYS = [
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]',
    ]
    session.preset_state = {
        'running': True,
        'total': len(rows),
        'failure_step': failure_step,
        'done': False,
    }

    for i, row_values in enumerate(rows):
        time.sleep(1)
        sensor_values = {k: v for k, v in zip(SENSOR_KEYS, row_values)}
        try:
            session.predict(sensor_values)
        except Exception as e:
            print(f"[run_preset_steps] Error at step {i + 1}: {e}")
            break

    session.preset_state['running'] = False
    session.preset_state['done'] = True


def get_log_files():
    """Returns sorted list of log filenames from the logs/ directory."""
    if not os.path.exists(LOGS_DIR):
        return []
    files = [f for f in os.listdir(LOGS_DIR) if f.endswith('.log')]
    files.sort(reverse=True)
    return files


def read_log_file(filename):
    """Reads a log file and returns its content."""
    path = os.path.join(LOGS_DIR, filename)
    if not os.path.exists(path):
        return "File not found."
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


# ── Dynamic model stats (parsed from most recent training log) ──────────────

def get_model_stats():
    """
    Returns a dict of model performance metrics parsed from the most recent
    main.py or model.py training log, plus live values from the loaded model.
    Falls back to None for any value that can't be parsed.
    """
    stats = {
        'roc_auc': None,
        'risk_tolerance': 0.765,
        'recall': None,
        'precision': None,
        'fpr': None,
        'tp': None,
        'fp': None,
        'fn': None,
        'tn': None,
        'feature_count': None,
        'log_file_used': None,
    }

    if not os.path.exists(LOGS_DIR):
        return stats

    # Collect all main.py and model.py log files, sorted most-recent first
    candidate_logs = sorted(
        [f for f in os.listdir(LOGS_DIR)
         if f.endswith('.log') and (f.startswith('main.py@') or f.startswith('model.py@'))],
        reverse=True
    )

    # Walk from most recent and pick the first one that contains training metrics
    for logfile in candidate_logs:
        path = os.path.join(LOGS_DIR, logfile)
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except OSError:
            continue

        # Only consider logs that have model evaluation output
        if 'ROC-AUC score' not in content:
            continue

        stats['log_file_used'] = logfile

        # ROC-AUC
        m = re.search(r'ROC-AUC score\s*:\s*([0-9.]+)', content)
        if m:
            stats['roc_auc'] = float(m.group(1))

        # Risk tolerance
        m = re.search(r'Risk Tolerance used\s*:\s*([0-9.]+)', content)
        if m:
            stats['risk_tolerance'] = float(m.group(1))

        # TP / FP
        m = re.search(r'TP:\s*(\d+)\s*\|\s*FP:\s*(\d+)', content)
        if m:
            stats['tp'] = int(m.group(1))
            stats['fp'] = int(m.group(2))

        # FN / TN
        m = re.search(r'FN:\s*(\d+)\s*\|\s*TN:\s*(\d+)', content)
        if m:
            stats['fn'] = int(m.group(1))
            stats['tn'] = int(m.group(2))

        # Recall
        m = re.search(r'Recall \(TPR\)\s*:\s*([0-9.]+)', content)
        if m:
            stats['recall'] = float(m.group(1))

        # Precision
        m = re.search(r'Precision\s*:\s*([0-9.]+)', content)
        if m:
            stats['precision'] = float(m.group(1))

        # FPR
        m = re.search(r'False PosRate\s*:\s*([0-9.]+)', content)
        if m:
            stats['fpr'] = float(m.group(1))

        break  # found a usable log — stop searching

    # Live feature count from model
    try:
        tmp_session = list(_sessions.values())[0] if _sessions else PredictionSession()
        stats['feature_count'] = len(tmp_session.risk_model.feature_names_in_)
    except Exception:
        pass

    return stats
