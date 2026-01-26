from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import auto_logger


if __name__ == '__main__':
    auto_logger.setup_logging('main.py')


class LogisticCassifier:
    def __init__(self):
        self.y = None
        self.x = None
        self.y_pred = None
        
    def dataset(self,target: str,ignore=list()):
        ignore += [target,]
        df = pd.read_csv('iot_equipment_monitoring_dataset.csv')
        self.y = df[target]
        self.X = df.drop(columns=ignore)
        print(*self.X.columns,sep='\n')
        
if __name__ == '__main__':
    model = LogisticCassifier()
    model.dataset('Fault_Type',['Timestamp','Sensor_ID'])
    