import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import shutil
import auto_logger


if __name__ == '__main__':
    auto_logger.setup_logging('IOT.py')

try:
    os.mkdir('Backup')
except:
    pass
file_path = "Backup/iot_equipment_monitoring_dataset.csv"

if not os.path.exists(file_path):
    shutil.copy2('iot_equipment_monitoring_dataset.csv',file_path)
    print(f"Backup made at {file_path}")
else:
    print('Backup exists proceeding noramlly....')


df=pd.read_csv('iot_equipment_monitoring_dataset.csv')
df['Fault_Type'] = df['Fault_Type'].fillna('None')
df.dropna()
pd.set_option('display.max_columns', None)
print(f'{'-'*100}\nTotal number of per columnDatapoints:\n{df.count()}\n{'-'*100}')
print(f'{'-'*100}\nAll numerical features of the columns:\n{df.describe()}\n{'-'*100}')

df["Fault_Type"] = df["Fault_Type"].fillna("None")

print(f'{'-'*100}\nCheecking for NaN values:\n{df.isna().mean().sort_values(ascending=False)}\n{'-'*100}')

print(f'{'-'*100}\ndtypes of the columns:\n{df.dtypes}\n{'-'*100}')

print(f'{df["Fault_Type"].value_counts(dropna=False)}')

plt.figure(figsize=(6, 4))
sns.countplot(
    data=df,
    x="Fault_Type",
    hue="Fault_Type",
    order=df["Fault_Type"].value_counts().index,
    palette='mako'
)
plt.title("Fault Type Frequency")
plt.show()

df.to_csv("iot_equipment_monitoring_dataset.csv", index=False)