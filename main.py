from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,confusion_matrix , ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import auto_logger
import gc

sns.set_theme(
    context="talk",   # larger fonts without shouting
    palette="crest_r"
)

if __name__ == '__main__':
    auto_logger.setup_logging('main.py')

try:
    os.mkdir('images')
except:
    pass

class LogisticClassifier:
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

        
    def train_run(self):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="lbfgs",
                class_weight="balanced",
                max_iter=3000
            ))
        ])

        param_dist = {
             "clf__C": np.logspace(-3, 2, 50)
        }
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=50,
            scoring="roc_auc",
            cv=5,
            random_state=42,
            n_jobs=-1,
            verbose = 1,
            error_score='raise'
        )
        search.fit(self.X_train,self.y_train)
        self.y_prob = search.predict_proba(self.X_test)[:, 1]
        self.roc = roc_auc_score(self.y_test, self.y_prob)
        #===== Tried Changing the Target Recall to clear my confusion =====
        #for i in [0.70,0.75,0.80,0.85]:
            #self.plot(self.roc,i)
        self.plot(self.roc,0.80)
    def plot(self,roc,target_recall):
        
        palette = sns.color_palette("crest", 10)

        # Dashboard layout
        layout = [["confusion", "roc"],
                ["prc", "kde"]]

        fig, axes = plt.subplot_mosaic(layout, figsize=(12, 10))

        #ROC AUC plot
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)

        axes["roc"].plot(fpr, tpr, lw=2.5, label=f"AUC = {roc:.3f}")
        axes["roc"].plot([0,1], [0,1], ls="--", color="gray", alpha=0.6)
        axes["roc"].fill_between(fpr, tpr, alpha=0.15)

        axes["roc"].set_xlabel("False Positive Rate")
        axes["roc"].set_ylabel("True Positive Rate")
        axes["roc"].set_title("ROC Curve")
        axes["roc"].legend()

        #KDE plot for Risk Score
        sns.kdeplot(
            self.y_prob[self.y_test == 0],
            fill=True,
            color=palette[1],
            alpha=0.6,
            linewidth=2,
            label="No Failure",
            ax=axes["kde"]
        )

        sns.kdeplot(
            self.y_prob[self.y_test == 1],
            fill=True,
            color=palette[7],
            alpha=0.6,
            linewidth=2,
            label="Failure",
            ax=axes["kde"]
        )

        axes["kde"].set_xlabel("Predicted Risk Score")
        axes["kde"].set_title("Risk Score Distribution")
        axes["kde"].legend()

        # Precision–Recall Curve
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_prob)
        ap = average_precision_score(self.y_test, self.y_prob)

        axes["prc"].plot(recall, precision, lw=2.5, label=f"AP = {ap:.3f}")
        axes["prc"].set_xlabel("Recall")
        axes["prc"].set_ylabel("Precision")
        axes["prc"].set_title("Precision–Recall Curve")
        axes["prc"].legend()

        #Confusion Matrix
        #target_recall = 0.80
        idx = np.where(recall >= target_recall)[0][-1]
        threshold = thresholds[idx]

        y_pred_thr = (self.y_prob >= threshold).astype(int)

        cm = confusion_matrix(self.y_test, y_pred_thr)
        tn, fp, fn, tp = cm.ravel()

        cm_percent = cm / cm.sum() * 100
        #labelling each cell of confusion matrix
        labels = np.array([
            [f"TN:{tn}\n({cm_percent[0,0]:.1f}%)", f"FP:{fp}\n({cm_percent[0,1]:.1f}%)"],
            [f"FN:{fn}\n({cm_percent[1,0]:.1f}%)", f"TP:{tp}\n({cm_percent[1,1]:.1f}%)"]
        ])

        sns.heatmap(
            cm / cm.max(),   # normalize colors
            annot=labels,
            fmt="",
            cmap="crest",
            linewidths=0.5,
            linecolor="white",
            xticklabels=["No Failure", "Failure"],
            yticklabels=["No Failure", "Failure"],
            alpha = 0.6,
            cbar=False,
            ax=axes["confusion"],
        )

        axes["confusion"].set_title(f"Confusion Matrix (threshold={threshold:.2f})")
        axes["confusion"].set_xlabel("Predicted")
        axes["confusion"].set_ylabel("Actual")
        axes["confusion"].grid(False)

        plt.tight_layout()
        fig.savefig(f"images/Dashboard_{self.filename}.png", dpi=150)
        plt.show()
        self.threshold = threshold
        self.y_pred_thr = y_pred_thr
        self.evaluate()
    def evaluate(self):
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred_thr).ravel()
        print("========== Model Features =========")
        print(*self.X.columns, sep='\n')
        print("\n=== Model Evaluation (Test Set) ===")
        print(f"ROC-AUC score :{self.roc:.3f}")
        print(f"Threshold used : {self.threshold:.3f}")
        print(f"TP: {tp} | FP: {fp}")
        print(f"FN: {fn} | TN: {tn}")

        print("\nRates:")
        print(f"Recall (TPR) : {tp / (tp + fn):.3f}")
        print(f"Precision    : {tp / (tp + fp):.3f}")
        print(f"False PosRate: {fp / (fp + tn):.3f}")
        
        print("====================================")
        gc.collect()
if __name__ == '__main__':
    model = LogisticClassifier()
    model.dataset('ai4i2020.csv','Machine failure',['TWF','HDF','PWF','OSF','RNF','UDI','Product ID','Type'])
    model.train_run()
    