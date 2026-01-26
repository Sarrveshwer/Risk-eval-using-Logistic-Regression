from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import auto_logger


if __name__ == '__main__':
    auto_logger.setup_logging('main.py')

try:
    os.mkdir('images')
except:
    pass

class LogisticCassifier:
    def __init__(self):
        self.y = None
        self.x = None
        self.y_pred = None
        self.y_prob = None
        self.filename = None
    def dataset(self,filename: str ,target: str, ignore=None):
        self.filename = filename.rstrip('.csv')
        if ignore is None:
            ignore = []

        df = pd.read_csv(filename)

        self.y = df[target].astype(int)
        self.X = df.drop(columns=ignore + [target])

        mask = self.X.notna().all(axis=1)
        self.X = self.X.loc[mask].reset_index(drop=True)
        self.y = self.y.loc[mask].reset_index(drop=True)

        idx = int(0.8 * len(self.X))

        self.X_train = self.X.iloc[:idx]
        self.y_train = self.y.iloc[:idx]

        self.X_test  = self.X.iloc[idx:]
        self.y_test  = self.y.iloc[idx:]

        print(*self.X.columns, sep='\n')
    def train_run(self):
        pipe = Pipeline(
            steps=[
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(
                        class_weight="balanced",
                        solver="liblinear",
                        max_iter=1000
                        ))
                    ]
                )
        pipe.fit(self.X_train,self.y_train)
        self.y_pred = pipe.predict(self.X_test)
        self.y_prob = pipe.predict_proba(self.X_test)[:, 1]
        roc = roc_auc_score(self.y_test, self.y_prob)
        print("ROC-AUC Score:", roc)
        self.plot(self.y,self.y_pred,roc)
    def plot(self,y,y_pred,roc):
        #ROC-AUC Plot
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)

        plt.figure(figsize=(5,5))
        plt.plot(fpr, tpr, label=f"ROC (AUC={roc:.3f})")
        plt.plot([0,1],[0,1],'--',color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.title("ROC Curve – Time-Aware Validation")
        plt.savefig(f'images/ROC-AUC_{self.filename}.png')
        plt.show()
        
        #Risk Score Distribution
        plt.figure(figsize=(6,4))
        sns.kdeplot(self.y_prob[self.y_test == 0], label="No Fault", fill=True)
        sns.kdeplot(self.y_prob[self.y_test == 1], label="Fault", fill=True)
        plt.xlabel("Predicted Risk Score")
        plt.title("Risk Score Distribution by Class")
        plt.legend()
        plt.savefig(f'images/Risk-Score-Distribution_{self.filename}.png')
        plt.show()
        
        #Precision Recall curve
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_prob)
        ap = average_precision_score(self.y_test, self.y_prob)

        plt.figure(figsize=(5,4))
        plt.plot(recall, precision, label=f"AP={ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve")
        plt.legend()
        plt.savefig(f'images/Precision-Recall-Curve_{self.filename}.png')
        plt.show()
if __name__ == '__main__':
    model = LogisticCassifier()
    model.dataset('iot_equipment_monitoring_dataset.csv','Fault_Status',['Timestamp','Sensor_ID','Fault_Type'])
    model.train_run()
    