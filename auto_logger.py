import sys
import os
import datetime
import traceback

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        self.is_newline = True  

    def write(self, message):
        self.terminal.write(message)
        if not self.log.closed:
            for char in message:
                if self.is_newline and char != '\n':
                    timestamp = datetime.datetime.now().strftime("[%H:%M:%S] ")
                    self.log.write(timestamp)
                    self.is_newline = False
                self.log.write(char)
                if char == '\n':
                    self.is_newline = True

    def flush(self):
        self.terminal.flush()
        if not self.log.closed:
            self.log.flush()

    def __del__(self):
        if hasattr(self, 'log') and not self.log.closed:
            self.log.close()

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print("\n" + "="*30 + " CRASH DETECTED " + "="*30)
    print(f"Timestamp: {datetime.datetime.now()}")
    print(error_msg)
    print("="*76 + "\n")

def setup_logging(script_name):
    try:
        os.mkdir("logs")
    except FileExistsError:
        pass
    
    safe_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{script_name}@{safe_timestamp}.log"
    log_path = os.path.join("logs", log_filename)
    
    # Redirect stdout
    sys.stdout = Logger(log_path)
    
    # Attach global exception hook
    sys.excepthook = handle_exception
    
    print(f"--- Logging Initialized for {script_name} ---")
    print(f"Log file: {log_path}")