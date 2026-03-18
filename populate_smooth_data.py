import random
import pymysql

# -- Config --
MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "root",
    "database": "ml_model",
}

def generate_smooth_data(n_rows=200, failure_onset=175):
    data = []
    
    # Normal baselines
    air_base = 300.3
    proc_base = 310.3
    rpm_base = 1550.0
    torque_base = 38.0
    wear_base = 50.0
    
    # Failure targets (Heat Failure style but slightly softened for prediction window)
    air_fail = 315.5
    proc_fail = 325.5
    rpm_fail = 1100.0
    torque_fail = 60.0
    wear_fail = 140.0

    for i in range(n_rows):
        if i < failure_onset:
            # Very stable operation with minimal jitter
            jitter = lambda: random.uniform(-0.05, 0.05)
            progress = i / failure_onset
            
            # Slow upward drift in temperatures and wear
            air = air_base + (progress * 0.4) + jitter()
            proc = proc_base + (progress * 0.4) + jitter()
            rpm = rpm_base + random.uniform(-5, 5)
            torque = torque_base + jitter()
            wear = wear_base + (i * 0.15) 
        else:
            # Smooth transition to failure
            p = (i - failure_onset) / (n_rows - failure_onset)
            s_curve = 3*p*p - 2*p*p*p 
            
            air = 300.7 + (air_fail - 300.7) * s_curve + random.uniform(-0.05, 0.05)
            proc = 310.7 + (proc_fail - 310.7) * s_curve + random.uniform(-0.05, 0.05)
            rpm = 1550 - (1550 - rpm_fail) * s_curve + random.uniform(-5, 5)
            torque = 38 + (torque_fail - 38) * s_curve + random.uniform(-0.2, 0.2)
            wear = 76 + (wear_fail - 76) * s_curve

        data.append((
            f"SIM-{i+1:03d}",
            round(air, 1),
            round(proc, 1),
            int(rpm),
            round(torque, 1),
            int(wear),
            0  # Machine failure = 0 for all
        ))
    return data

def main():
    rows = generate_smooth_data()
    table_name = "smooth_simulation"
    
    print(f"Connecting to MySQL at {MYSQL_CONFIG['host']}...")
    try:
        conn = pymysql.connect(**MYSQL_CONFIG, connect_timeout=5)
        print("Connected.")
        with conn.cursor() as cursor:
            # Use backticks for table and column names with spaces
            cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")
            cursor.execute(f"""
                CREATE TABLE `{table_name}` (
                    `Product ID` VARCHAR(50),
                    `Air temperature [K]` FLOAT,
                    `Process temperature [K]` FLOAT,
                    `Rotational speed [rpm]` INT,
                    `Torque [Nm]` FLOAT,
                    `Tool wear [min]` INT,
                    `Machine failure` INT
                )
            """)
            
            sql = f"INSERT INTO `{table_name}` " \
                  "(`Product ID`, `Air temperature [K]`, `Process temperature [K]`, " \
                  "`Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `Machine failure`) " \
                  "VALUES (%s, %s, %s, %s, %s, %s, %s)"
            
            cursor.executemany(sql, rows)
            conn.commit()
            print(f"Successfully created table `{table_name}` with {len(rows)} rows.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'conn' in locals() and conn.open:
            conn.close()

if __name__ == "__main__":
    main()
