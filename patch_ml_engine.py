import re

with open('dashboard/predictor/ml_engine.py', 'r') as f:
    text = f.read()

# We need to change both lines 256 and 267 where ORDER BY RAND() is used since we have no sequence info
# We should drop the random MySQL fetch entirely for the specific failure tests and just let it fall back 
# to the synthetic dynamic logic which DOES generate a perfect gradual failure trend.

new_text = text.replace('db_rows = _fetch_db_rows(preset_name)', '''
    # Do not fetch from MySQL for the predefined failure tests, so the synthetic 
    # dynamic failure buildup (ramp-up) works properly.
    if preset_name in ["hdf", "twf", "osf", "pwf", "random_failure"]:
        db_rows = None
    else:
        db_rows = _fetch_db_rows(preset_name)
''')

with open('dashboard/predictor/ml_engine.py', 'w') as f:
    f.write(new_text)
print("Patched.")
