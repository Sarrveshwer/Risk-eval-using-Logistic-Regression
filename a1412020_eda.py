import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import shutil
import auto_logger


if __name__ == '__main__':
    auto_logger.setup_logging('a1412020.py')

try:
    os.mkdir('Backup')
except:
    pass
try:
    os.mkdir('images')
except:
    pass

file_path = "Backup/ai4i2020.csv"

if not os.path.exists(file_path):
    shutil.copy2('ai4i2020.csv',file_path)
    print(f"Backup made at {file_path}")
else:
    print('Backup exists proceeding noramlly....')


df=pd.read_csv('ai4i2020.csv')
pd.set_option('display.max_columns', None)
print(f'{'-'*100}\nTotal number of per columnDatapoints:\n{df.count()}\n{'-'*100}')
print(f'{'-'*100}\nAll numerical features of the columns:\n{df.describe()}\n{'-'*100}')


print(f'{'-'*100}\nCheecking for NaN values:\n{df.isna().mean().sort_values(ascending=False)}\n{'-'*100}')

print(f'{'-'*100}\ndtypes of the columns:\n{df.dtypes}\n{'-'*100}')

plt.figure(figsize=(6, 4))
ax = sns.countplot(
    data=df,
    x="Machine failure",
    hue = "Machine failure",
    order=df["Machine failure"].value_counts().index,
    palette="mako"
)

# Add counts on top of bars
for container in ax.containers:
    ax.bar_label(container, padding=3)
    
plt.title("Machine failure Frequency")
plt.savefig('images/Machine failure Frequency.png')
plt.show()

df.to_csv("iot_equipment_monitoring_dataset.csv", index=False)