from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix , ConfusionMatrixDisplay
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


# ─── Risk Finding Model ───────────────────────────────────────────────────────


'''
This model uses LogisticRegression to give me a prelimnary prediction whether 
the machine is going to fail or not.
'''
class RiskEvalModel:
    def __init__(self,filename,target,risk_tolerance,ignore=None):
        filename_modified, ext = os.path.splitext(filename)
        self.ext = ext
        self.df = None
        self.y = None
        self.x = None 
        self.risk_tolerance = risk_tolerance
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
        '''
        Created a pipline where StandardScaler[z-scalling] is first applied and then
        piped to Logistic Regression for model fitting
        '''
        self.X,self.y,self.X_train,self.y_train,self.X_test,self.y_test,self.df=self.dataset(self.filename+self.ext,self.target,False,self.ignore)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=3000
            ))
        ])

        param_dist = {
            "clf__C": np.logspace(-5, -1, 50) # Range shifted from -3:2 to -5:-1
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
        self.plot(self.roc,self.risk_tolerance)
        model_name =  "Risk_eval_model.pkl"
        jb.dump(search,model_name)
        
        return model_name
    
    def plot(self,roc,target_recall):
        '''
        thsis method is for making 4 different graphs integrated into a single dashboard
        1) KDE plot of risk
        2) Confusion Matrix
        3) ROC-AUC graph
        4) Precision-Recall Graph
        '''
        palette = sns.color_palette("crest", 10)

        layout = [["confusion", "roc"],
                ["prc", "kde"]]

        fig, axes = plt.subplot_mosaic(layout, figsize=(12, 10))

        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)

        axes["roc"].plot(fpr, tpr, lw=2.5, label=f"AUC = {roc:.3f}")
        axes["roc"].plot([0,1], [0,1], ls="--", color="gray", alpha=0.6)
        axes["roc"].fill_between(fpr, tpr, alpha=0.15)

        axes["roc"].set_xlabel("False Positive Rate")
        axes["roc"].set_ylabel("True Positive Rate")
        axes["roc"].set_title("ROC Curve")
        axes["roc"].legend()

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

        precision, recall, risk_tolerances = precision_recall_curve(self.y_test, self.y_prob)
        ap = average_precision_score(self.y_test, self.y_prob)

        axes["prc"].plot(recall, precision, lw=2.5, label=f"AP = {ap:.3f}")
        axes["prc"].set_xlabel("Recall")
        axes["prc"].set_ylabel("Precision")
        axes["prc"].set_title("Precision–Recall Curve")
        axes["prc"].legend()

        idx = np.where(recall >= target_recall)[0][-1]
        risk_tolerance = risk_tolerances[idx]

        y_pred_thr = (self.y_prob >= risk_tolerance).astype(int)

        cm = confusion_matrix(self.y_test, y_pred_thr)
        tn, fp, fn, tp = cm.ravel()

        cm_percent = cm / cm.sum() * 100
        labels = np.array([
            [f"TN:{tn}\n({cm_percent[0,0]:.1f}%)", f"FP:{fp}\n({cm_percent[0,1]:.1f}%)"],
            [f"FN:{fn}\n({cm_percent[1,0]:.1f}%)", f"TP:{tp}\n({cm_percent[1,1]:.1f}%)"]
        ])

        sns.heatmap(
            cm / cm.max(),
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

        axes["confusion"].set_title(f"Confusion Matrix (risk_tolerance={risk_tolerance:.2f})")
        axes["confusion"].set_xlabel("Predicted")
        axes["confusion"].set_ylabel("Actual")
        axes["confusion"].grid(False)

        plt.tight_layout()
        fig.savefig(f"images/Dashboard_{self.filename}.png", dpi=300)
        plt.show()
        self.risk_tolerance = risk_tolerance
        self.y_pred_thr = y_pred_thr
        self.evaluate()
        
    def evaluate(self):
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred_thr).ravel()
        print("========== Model Features =========")
        print(*self.X.columns, sep='\n')
        print("\n=== Model Evaluation (Test Set) ===")
        print(f"ROC-AUC score :{self.roc:.3f}")
        print(f"Risk Tolerance used : {self.risk_tolerance:.3f}")
        print(f"TP: {tp} | FP: {fp}")
        print(f"FN: {fn} | TN: {tn}")

        print("\nRates:")
        print(f"Recall (TPR) : {tp / (tp + fn):.3f}")
        print(f"Precision    : {tp / (tp + fp):.3f}")
        print(f"False PosRate: {fp / (fp + tn):.3f}")
        
        print("====================================")
        gc.collect()

# ─── Failure Classification Model ─────────────────────────────────────────────

class FailureClassificationModel:
    def __init__(self,filename,target,risk_tolerance,ignore=None,classifications = None):
        filename_modified, ext = os.path.splitext(filename)
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
        if ignore == None:
            self.ignore = []
        else:
            self.ignore = ignore
        
        del ignore
        del filename_modified
        del ext
        
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
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=3000))
        ])

        pipe.fit(self.X_train,self.y_train)
        self.y_prob = pipe.predict_proba(self.X_test)[:, 1]
        self.roc = roc_auc_score(self.y_test, self.y_prob)
        self.plot(self.roc,0.50,param)
        self.evaluate()
        
        model_name= f"Risk_eval_{param}.pkl"
        jb.dump(pipe,model_name)
        
        return model_name
    
    def plot(self,roc,target_recall,param):
        
        palette = sns.color_palette("crest", 10)

        layout = [["confusion", "roc"],
                ["prc", "kde"]]

        fig, axes = plt.subplot_mosaic(layout, figsize=(12, 10))

        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)

        axes["roc"].plot(fpr, tpr, lw=2.5, label=f"AUC = {roc:.3f}")
        axes["roc"].plot([0,1], [0,1], ls="--", color="gray", alpha=0.6)
        axes["roc"].fill_between(fpr, tpr, alpha=0.15)

        axes["roc"].set_xlabel("False Positive Rate")
        axes["roc"].set_ylabel("True Positive Rate")
        axes["roc"].set_title("ROC Curve")
        axes["roc"].legend()

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

        precision, recall, risk_tolerances = precision_recall_curve(self.y_test, self.y_prob)
        ap = average_precision_score(self.y_test, self.y_prob)

        axes["prc"].plot(recall, precision, lw=2.5, label=f"AP = {ap:.3f}")
        axes["prc"].set_xlabel("Recall")
        axes["prc"].set_ylabel("Precision")
        axes["prc"].set_title("Precision–Recall Curve")
        axes["prc"].legend()

        idx = np.where(recall >= target_recall)[0][-1]
        self.risk_tolerance = risk_tolerances[idx]

        self.y_pred_thr = (self.y_prob >= self.risk_tolerance).astype(int)

        cm = confusion_matrix(self.y_test, self.y_pred_thr)
        tn, fp, fn, tp = cm.ravel()

        cm_percent = cm / cm.sum() * 100
        labels = np.array([
            [f"TN:{tn}\n({cm_percent[0,0]:.1f}%)", f"FP:{fp}\n({cm_percent[0,1]:.1f}%)"],
            [f"FN:{fn}\n({cm_percent[1,0]:.1f}%)", f"TP:{tp}\n({cm_percent[1,1]:.1f}%)"]
        ])

        sns.heatmap(
            cm / cm.max(),
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

        axes["confusion"].set_title(f"Confusion Matrix (self.risk_tolerance={self.risk_tolerance:.2f})")
        axes["confusion"].set_xlabel("Predicted")
        axes["confusion"].set_ylabel("Actual")
        axes["confusion"].grid(False)

        plt.tight_layout()
        fig.savefig(f"images/Dashboard_{param}_{self.filename}.png", dpi=300)
        plt.show()
        self.evaluate()
        
    def evaluate(self):
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred_thr).ravel()
        print("========== Model Features =========")
        print(*self.X.columns, sep='\n')
        print("\n=== Model Evaluation (Test Set) ===")
        print(f"ROC-AUC score :{self.roc:.3f}")
        print(f"self.risk_tolerance used : {self.risk_tolerance:.3f}")
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
 
# ─── Complete System ──────────────────────────────────────────────────────────

'''
The below class encorporates both models first the RiskEvalModel runs if the 
model detects a risk FailureClassificationModel kicks in to find what type of 
failure occured.
'''
    
class FailurePredictionSystem:
    def __init__(self, filename, target, risk_tolerance, ignore=None, classifications=None, 
                 warning_sensitivity=0.6, diagnosis_sensitivity=0.4, persistence_threshold=3):
        self.filename = filename
        self.target = target
        self.risk_tolerance = risk_tolerance
        self.classifications = classifications
        self.ignore = ignore if ignore else []
        self.history = []
        self.risk_model = None
        self.classification_models = {}
        self.warning_streak = 0
        self.warning_sensitivity = warning_sensitivity
        self.diagnosis_sensitivity = diagnosis_sensitivity
        self.persistence_threshold = persistence_threshold

    def LoadModels(self):
        risk_model_path = "Risk_eval_model.pkl"
        if os.path.exists(risk_model_path):
            self.risk_model = jb.load(risk_model_path)
            print("Loaded RiskEvalModel from disk.")
        else:
            risk_model = RiskEvalModel(
                self.filename,
                self.target,
                self.risk_tolerance,
                self.ignore + self.classifications
            )
            risk_model_path = risk_model.train_run()
            self.risk_model = jb.load(risk_model_path)

        for failure in self.classifications:
            model_path = f"Risk_eval_{failure}.pkl"
            if os.path.exists(model_path):
                self.classification_models[failure] = jb.load(model_path)
            else:
                clf = FailureClassificationModel(
                    self.filename,
                    self.target,
                    self.risk_tolerance,
                    self.ignore,
                    self.classifications
                )
                model_path = clf.train_run(failure)
                self.classification_models[failure] = jb.load(model_path)

    def predict(self, X: pd.DataFrame):
        current_row_dict = X.iloc[0].to_dict()
        self.history.append(current_row_dict)
        if len(self.history) > 5:
            self.history.pop(0)

        current_data_df = X.copy()
        history_dataframe = pd.DataFrame(self.history)
        
        cols_to_exclude = [self.target] + self.classifications + self.ignore
        
        dynamic_features = [
            c for c in X.columns 
            if c not in cols_to_exclude 
            and not any(x in c for x in ['_Rolling_Mean', '_Volatility', '_Delta', '_Rolling_Delta'])
        ]

        for col in dynamic_features:
            current_data_df[f'{col}_Rolling_Mean'] = history_dataframe[col].mean()
            
            if len(self.history) > 1:
                current_data_df[f'{col}_Volatility'] = history_dataframe[col].std()
                current_data_df[f'{col}_Delta'] = self.history[-1][col] - self.history[-2][col]
                current_data_df[f'{col}_Rolling_Delta'] = history_dataframe[col].diff().mean()
            else:
                current_data_df[f'{col}_Volatility'] = 0.0
                current_data_df[f'{col}_Delta'] = 0.0
                current_data_df[f'{col}_Rolling_Delta'] = 0.0

        current_data_df = current_data_df.fillna(0)

        inference_df = current_data_df.drop(columns=[c for c in cols_to_exclude if c in current_data_df.columns])
        inference_df = inference_df[self.risk_model.feature_names_in_]
        
        raw_risk_prob = float(self.risk_model.predict_proba(inference_df)[:, 1][0])
        
        if not hasattr(self, 'prob_history'):
            self.prob_history = []
            
        self.prob_history.append(raw_risk_prob)
        if len(self.prob_history) > 3: 
            self.prob_history.pop(0)

        risk_prob = sum(self.prob_history) / len(self.prob_history)
        
        warning_zone = self.risk_tolerance * self.warning_sensitivity
        
        if risk_prob >= warning_zone and risk_prob < self.risk_tolerance:
            self.warning_streak += 1
        elif risk_prob >= self.risk_tolerance:
            self.warning_streak = 0
        else:
            self.warning_streak = 0

        if risk_prob >= self.risk_tolerance:
            alert_status = "CRITICAL"
        elif risk_prob >= warning_zone:
            self.warning_streak += 1
            if self.warning_streak >= self.persistence_threshold:
                alert_status = "WARNING"
            else:
                alert_status = "HEALTHY"
        else:
            alert_status = "HEALTHY"

        results = {name: round(float(m.predict_proba(inference_df)[:, 1][0]), 4) 
                   for name, m in self.classification_models.items()}
        
        active = [f for f, p in results.items() if p >= self.diagnosis_sensitivity]
        failure = max(results, key=results.get) if active else "RandomFailure"

        return {
            "risk_prob": round(risk_prob, 4), 
            "alert_level": alert_status, 
            "failure_type": failure if alert_status != "HEALTHY" else None,
            "streak": self.warning_streak,
            "details": results
        }
        
    def evaluate_test_set(self, test_dataframe_X):
        print(f"{'Step':<5} | {'Risk':<7} | {'Streak':<6} | {'Alert':<10} | {'Diagnosis'}")
        print("-" * 65)
        for i in range(len(test_dataframe_X)):
            res = self.predict(test_dataframe_X.iloc[[i]])
            print(f"{i+1:<5} | {res['risk_prob']:<7.4f} | {res['streak']:<6} | {res['alert_level']:<10} | {res['failure_type']}")
    def calibrate_sensitivities(self, test_df, target_step_for_warning=4):
            best_score = -1
            best_params = {}
            
            warning_range = np.linspace(0.3, 0.7, 5)
            diagnosis_range = np.linspace(0.2, 0.5, 4)
            
            print(f"{'Warning_S':<10} | {'Diag_S':<10} | {'Warning Step':<12} | {'Score'}")
            print("-" * 50)
            
            for w_s in warning_range:
                for d_s in diagnosis_range:
                    self.warning_sensitivity = w_s
                    self.diagnosis_sensitivity = d_s
                    self.warning_streak = 0
                    self.history = []
                    
                    first_warning_step = None
                    
                    for i in range(len(test_df)):
                        res = self.predict(test_df.iloc[[i]])
                        if res['alert_level'] != "HEALTHY" and first_warning_step is None:
                            first_warning_step = i + 1
                    
                    if first_warning_step:
                        score = 1.0 / (abs(target_step_for_warning - first_warning_step) + 1)
                    else:
                        score = 0
                    
                    print(f"{w_s:<10.2f} | {d_s:<10.2f} | {str(first_warning_step):<12} | {score:.2f}")
                    
                    if score > best_score:
                        best_score = score
                        best_params = {'warning_sensitivity': w_s, 'diagnosis_sensitivity': d_s}
            
            print("-" * 50)
            print(f"Optimal Parameters: {best_params}")
            return best_params
        
if __name__ == '__main__':
    target_labels = ['TWF','HDF','PWF','OSF']
    ignore_list = ['RNF','UDI','Product ID','Type'] + target_labels
    
    FinalSystem = FailurePredictionSystem(
        'ai4i2020.csv',
        'Machine failure',
        0.80,
        ignore_list,
        target_labels
    )
    FinalSystem.LoadModels()

    test_rows = [
        [300.1, 310.2, 1500, 40.0, 100], 
        [300.4, 310.5, 1500, 40.2, 102], 
        [301.2, 311.0, 1500, 40.5, 104], 
        [302.5, 312.1, 1500, 41.0, 106], 
        [303.8, 313.4, 1500, 41.5, 108], 
        [304.9, 314.5, 1500, 42.0, 110],
        [305.5, 315.2, 1500, 42.5, 112], 
        [306.1, 315.8, 1500, 43.0, 114], 
        [308.5, 318.0, 1500, 45.0, 116], 
        [312.0, 322.0, 1000, 65.0, 120]  
    ]
    cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    test_data = pd.DataFrame(test_rows, columns=cols)

    best_config = FinalSystem.calibrate_sensitivities(test_data, target_step_for_warning=4)
    
    FinalSystem.warning_sensitivity = best_config['warning_sensitivity']
    FinalSystem.diagnosis_sensitivity = best_config['diagnosis_sensitivity']
    
    FinalSystem.evaluate_test_set(test_data)