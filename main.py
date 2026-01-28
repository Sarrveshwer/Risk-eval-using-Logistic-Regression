from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,confusion_matrix , ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import joblib as jb
import numpy as np
import os
import auto_logger
import gc

sns.set_theme(
    context="talk",   
    palette="crest_r"
)

if __name__ == '__main__':
    auto_logger.setup_logging('main.py')

try:
    os.mkdir('images')
except:
    pass

class RiskEvalModel:
    def __init__(self,filename,target,threshold,ignore=None):
        filename_modified, ext = os.path.splitext(filename)
        self.ext = ext
        self.df = None
        self.y = None
        self.x = None 
        self.threshold = threshold
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        self.y_pred = None
        self.y_prob = None
        self.filename = filename_modified
        self.target = target
        if ignore == None:
            self.ignore = []
        else:
            self.ignore = ignore
        del ignore
        del ext 
        del filename_modified
        
    def dataset(self, filename: str,target: str,failure_only = False,ignore=None):
        if ignore is None:
            ignore = []

        df = pd.read_csv(filename)

        if failure_only:
            df = df[df[target] == 1]
        
        y = df[target].astype(int)
        X = df.drop(columns=ignore + [target])

        mask = X.notna().all(axis=1)
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)

        idx = int(0.8 * len(X))

        X_train = X.iloc[:idx]
        y_train = y.iloc[:idx]

        X_test  = X.iloc[idx:]
        y_test  = y.iloc[idx:]
        
        return X,y,X_train,y_train,X_test,y_test,df

    def train_run(self):
        
        self.X,self.y,self.X_train,self.y_train,self.X_test,self.y_test,self.df=self.dataset(self.filename+self.ext,self.target,False,self.ignore)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
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
        self.plot(self.roc,self.threshold)
        model_name =  "Risk_eval_model.pkl"
        jb.dump(search,model_name)
        
        return model_name
    
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
        fig.savefig(f"images/Dashboard_{self.filename}.png", dpi=300)
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
        
class FailureClassificationModel:
    def __init__(self,filename,target,threshold,ignore=None,classifications = None):
        filename_modified, ext = os.path.splitext(filename)
        #initializing all variables used accross the class
        self.ext = ext
        self.df = None
        self.y = None
        self.x = None 
        self.X_test = None
        self.y_test = None
        self.X_train = None
        self.y_train = None
        self.y_pred = None
        self.y_prob = None
        self.filename = filename_modified
        self.target = target
        self.classifications = classifications
        #chaning ignore from None to [] if ignore == None
        if ignore == None:
            self.ignore = []
        else:
            #If ignore not [] then save the list across the class using self.
            self.ignore = ignore
        
        del ignore
        del filename_modified
        del ext
        
        #Checks if the Classification argument passed is empty and raises Error according
        if self.classifications == None or self.classifications == []:
            print('Error classification cannot be empty....')
        
    def dataset(self, filename: str,target: str,filter: str,failure_only = False,ignore=None):
        if ignore is None:
            ignore = []

        df = pd.read_csv(filename)

        df = df[df[filter] == 1]
        
        y = df[target].astype(int)
        X = df.drop(columns=ignore + [target])

        mask = X.notna().all(axis=1)
        self.X = X.loc[mask].reset_index(drop=True)
        self.y = y.loc[mask].reset_index(drop=True)

        idx = int(0.8 * len(X))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                                            self.X, self.y, 
                                                            test_size=0.2, 
                                                            stratify=y,
                                                            random_state=42
                                                        )
        

    def train_run(self,param):
        remaining_params = [i for i in self.classifications if i != param]
        self.dataset(self.filename + self.ext,
                    param,
                    self.target,
                    True,
                    self.ignore + remaining_params + [self.target]
                    )
        rfc = RandomForestClassifier()

        rfc.fit(self.X_train,self.y_train)
        self.y_prob = rfc.predict_proba(self.X_test)[:, 1]
        self.roc = roc_auc_score(self.y_test, self.y_prob)
        self.plot(self.roc,0.80,param)
        self.evaluate()
        
        model_name= f"Risk_eval_{param}.pkl"
        jb.dump(rfc,model_name)
        
        return model_name
    
    def plot(self,roc,target_recall,param):
        
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
        fig.savefig(f"images/Dashboard_{param}_{self.filename}.png", dpi=300)
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
        
    def run(self):
        model_names = []
        for i in self.classifications:
            model_names.append(self.train_run(i))
        return model_names
class FailurePredictionSystem:
    def __init__(self, filename, target, threshold, ignore=None, classifications=None):
        self.filename = filename
        self.target = target
        self.threshold = threshold
        self.classifications = classifications

        if ignore is None:
            self.ignore = []
        else:
            self.ignore = ignore

        if not self.classifications:
            raise ValueError("classification list cannot be empty")

        self.risk_model = None
        self.classification_models = {}

    def LoadModels(self):
        # ---- Risk model ----
        risk_model_path = "Risk_eval_model.pkl"

        if os.path.exists(risk_model_path):
            self.risk_model = jb.load(risk_model_path)
            print("Loaded RiskEvalModel from disk.")
        else:
            print("Training RiskEvalModel...")
            risk_model = RiskEvalModel(
                self.filename,
                self.target,
                self.threshold,
                self.ignore + self.classifications
            )
            risk_model_path = risk_model.train_run()
            self.risk_model = jb.load(risk_model_path)

        self.classification_models = {}

        for failure in self.classifications:
            model_path = f"Risk_eval_{failure}.pkl"

            if os.path.exists(model_path):
                self.classification_models[failure] = jb.load(model_path)
                print(f"Loaded classifier for {failure}")
            else:
                print(f"Training classifier for {failure}...")
                clf = FailureClassificationModel(
                    self.filename,
                    self.target,
                    self.threshold,
                    self.ignore,
                    self.classifications
                )
                model_path = clf.train_run(failure)
                self.classification_models[failure] = jb.load(model_path)

        print("All models ready.")

    def predict(self, X: pd.DataFrame):
        # enforce single-row inference
        if len(X) != 1:
            X = X.iloc[[0]]

        risk_prob = float(self.risk_model.predict_proba(X)[:, 1][0])

        if risk_prob <= self.threshold:
            return {
                "risk_prob": risk_prob,
                "risk": 0,
                "failure_type": None
            }

        results = {}

        for name, model in self.classification_models.items():
            prob = float(model.predict_proba(X)[:, 1][0])
            results[name] = int(prob >= 0.5)

        # RNF logic
        if all(v == 0 for v in results.values()):
            failure = "RandomFailure"
        else:
            failure = max(results, key=results.get)

        return {
            "risk_prob": risk_prob,
            "risk": 1,
            "failure_type": failure,
            "details": results
        }


if __name__ == '__main__':
    model = FailurePredictionSystem('ai4i2020.csv',
                                    'Machine failure',
                                   0.80,
                                   ['RNF','UDI','Product ID','Type'],
                                   ['TWF','HDF','PWF','OSF']
                                   )
    moedl_names = model.LoadModels()
    df = pd.read_csv("ai4i2020.csv")

    X = df.drop(
        columns=[
            "Machine failure",
            "TWF",
            "HDF",
            "PWF",
            "OSF",
            "RNF",
            "UDI",
            "Product ID",
            "Type"
        ],
        errors="ignore"
    ).iloc[:5]
    predictions = model.predict(X)
    print(predictions)
    
