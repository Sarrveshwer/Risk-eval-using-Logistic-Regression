import os
import datetime
import pymysql
import pandas as pd

# ── Unified Configuration ───────────────────────────────────────────────────
# These serve as the system-wide defaults, but can be overridden 
# by passing credentials to get_connection() or other functions.
MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "root",
    "database": "ml_model",  # Dashboard test tables
}

SOURCE_DATABASE = "ml_model" # Training data database
SOURCE_TABLE = "ai4i2020"

# Find log directory relative to the project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FRONT_LOGS_DIR = os.path.join(ROOT_DIR, "log_front")

# ── Connection Factory ──────────────────────────────────────────────────────

def get_connection(user=None, password=None, database=None, host=None):
    """
    Returns a PyMySQL connection. Uses defaults if parameters are missing.
    In Production, the Dashboard passes the session's username/password here.
    """
    config = MYSQL_CONFIG.copy()
    if user: config["user"] = user
    if password: config["password"] = password
    if database: config["database"] = database
    if host: config["host"] = host
    
    return pymysql.connect(**config)

# ── Data Operations ─────────────────────────────────────────────────────────

def load_training_df(user=None, password=None, database=SOURCE_DATABASE):
    """Loads the entire training dataset (e.g., ai4i2020 from ml_data)."""
    conn = get_connection(user=user, password=password, database=database)
    try:
        df = pd.read_sql(f"SELECT * FROM `{SOURCE_TABLE}`", conn)
        print(f"[DataLayer] Loaded {len(df):,} rows from '{database}.{SOURCE_TABLE}'")
        return df
    finally:
        conn.close()

def fetch_simulation_rows(preset_name, user=None, password=None, host=None, database=None):
    """
    Fetches raw simulation rows from specialized test tables 
    (test_normal, test_hdf, etc.).
    """
    table_map = {
        "normal": "test_normal",
        "hdf": "test_hdf",
        "twf": "test_twf",
        "osf": "test_osf",
        "pwf": "test_pwf",
        "random_failure": "test_random_failure",
        "database": "smooth_simulation",
    }
    table = table_map.get(preset_name, "test_normal")
    
    conn = get_connection(user=user, password=password, host=host, database=database)
    try:
        count_df = pd.read_sql(f"SELECT COUNT(*) as cnt FROM `{table}`", conn)
        total_rows = count_df.iloc[0]['cnt']
        
        offset = 0
        if total_rows > 20:
            import random
            offset = random.randint(0, int(total_rows) - 20)
            
        df = pd.read_sql(f"SELECT * FROM `{table}` LIMIT 20 OFFSET {offset}", conn)
        sensor_cols = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]"
        ]
        return df[sensor_cols].values.tolist()
    finally:
        conn.close()

# ── Logging Operations ──────────────────────────────────────────────────────

def get_test_log_path():
    """Returns the path for today's log file (YYYY-MM-DD.log)."""
    os.makedirs(FRONT_LOGS_DIR, exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    return os.path.join(FRONT_LOGS_DIR, f"{date_str}.log")

def write_test_log(line: str):
    """Appends a line with timestamp to the current day's log file."""
    path = get_test_log_path()
    timestamp = datetime.datetime.now().strftime("[%H:%M:%S] ")
    with open(path, "a", encoding="utf-8") as f:
        f.write(timestamp + line + "\n")

def get_log_files():
    """Returns a sorted list of all .log filenames in the log directory."""
    if not os.path.exists(FRONT_LOGS_DIR):
        return []
    files = [f for f in os.listdir(FRONT_LOGS_DIR) if f.endswith(".log")]
    return sorted(files, reverse=True)

def read_log_file(filename):
    """Reads the content of a specific log file by name."""
    # Security: ensure only .log files in FRONT_LOGS_DIR can be read
    if not filename.endswith(".log"):
        return "Error: Invalid file type."
    
    path = os.path.join(FRONT_LOGS_DIR, os.path.basename(filename))
    if not os.path.exists(path):
        return f"Error: File {filename} not found."
    
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ── Auth Operations ──────────────────────────────────────────────────────────

def validate_credentials(user, password, host=None, database=None):
    """
    Validates MySQL credentials by attempting a connection.
    Used by the Login system.
    """
    try:
        conn = get_connection(user=user, password=password, host=host, database=database)
        conn.close()
        return True, None
    except pymysql.err.OperationalError as e:
        return False, f"Authentication failed: {str(e)}"
    except Exception as e:
        return False, f"Connection error: {str(e)}"
