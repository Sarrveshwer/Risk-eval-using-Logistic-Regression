from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix , ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import joblib as jb
import numpy as np
import os
import auto_logger
import gc
from sqlalchemy import create_engine

# ─── MySQL connection config ───────────────────────────────────────────────────
MYSQL_CONFIG = {
    'host':     'localhost',
    'user':     'root',
    'password': 'root',
    'database': 'ml_data',
}
SOURCE_TABLE = 'ai4i2020'


def load_dataframe() -> pd.DataFrame:
    '''
    Loads the full training dataset directly from MySQL.
    Returns a pandas DataFrame identical to what pd.read_csv('ai4i2020.csv') used to return.
    '''
    url = (
        f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}"
        f"@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}"
    )
    print(f"[MySQL] Connecting to {MYSQL_CONFIG['user']}@{MYSQL_CONFIG['host']}/{MYSQL_CONFIG['database']}")
    engine = create_engine(url)
    with engine.connect() as conn:
        df = pd.read_sql(f"SELECT * FROM `{SOURCE_TABLE}`", conn)
    print(f"[MySQL] Loaded {len(df):,} rows from '{SOURCE_TABLE}'")
    return df

sns.set_theme(
    context="talk",   
    palette="crest_r"
)

# ─── Sets Up Automatic Logging ────────────────────────────────────────────────

if __name__ == '__main__':
    auto_logger.setup_logging('model.py')

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
    def __init__(self, df: pd.DataFrame, target, risk_tolerance, ignore=None):
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
        self.filename = 'ai4i2020'     # kept for image-save filenames
        self.source_df = df            # the already-loaded DataFrame
        self.target = target
        if ignore is None:
            self.ignore = []
        else:
            self.ignore = ignore
    
    # ─── Import The Dataset And Split The Data In Two ─────────────────────
    
    def dataset(self, target: str, failure_only=False, ignore=None):
        '''
        This function reads the dataset and splits the dataset into training and testing data.
        Data comes from the already-loaded self.source_df (MySQL), not a CSV.
        '''
        if ignore is None:
            ignore = []

        self.df = self.source_df.copy()

        if failure_only:
            self.df = self.df[self.df[target] == 1]
        
        y = self.df[target].astype(int)
        X = self.df.drop(columns=ignore + [target])

        mask = X.notna().all(axis=1)
        self.X = X.loc[mask].reset_index(drop=True)
        self.y = y.loc[mask].reset_index(drop=True)

        idx = int(0.8 * len(self.X))

        self.X_train = self.X.iloc[:idx]
        self.y_train = self.y.iloc[:idx]

        self.X_test  = self.X.iloc[idx:]
        self.y_test  = self.y.iloc[idx:]
        
    # ─── Train The Model And Run A Trial ──────────────────────────────────

    
    def train_run(self):
        '''
        Created a pipline where RobustScaler is first applied and then
        piped to Logistic Regression for model fitting
        '''
        
        self.dataset(self.target, False, self.ignore)
        
        #Creates the pipline for the model to work
        pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                penalty='l1',      # Pure L1
                solver='liblinear', # Supports L1 for binary
                max_iter=3000
            ))
        ])

        param_dist = {
            "clf__C": np.logspace(-5, 7 , 50) 
        }
        
        #Finds the best paramater for LogisticRegression
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=50,
            scoring="roc_auc",
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose = 1,
            error_score='raise'
        )
        
        #Fitting the data into the model
        search.fit(self.X_train,self.y_train)
        self.y_prob = search.predict_proba(self.X_test)[:, 1]
        self.roc = roc_auc_score(self.y_test, self.y_prob)
        self.plot(self.roc,self.risk_tolerance)
        model_name =  "models/Risk_eval_model.pkl"
        jb.dump(search, model_name)
        
        return model_name
    
    # ─── Plot All The Graphs Related To The Model ─────────────────────────

    
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

        #findint FPR and TPR to be plotted in the ROC-AUC plot
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)

        #plots the ROC-AUC plot
        axes["roc"].plot(fpr, tpr, lw=2.5, label=f"AUC = {roc:.3f}")
        axes["roc"].plot([0,1], [0,1], ls="--", color="gray", alpha=0.6)
        axes["roc"].fill_between(fpr, tpr, alpha=0.15)

        axes["roc"].set_xlabel("False Positive Rate")
        axes["roc"].set_ylabel("True Positive Rate")
        axes["roc"].set_title("ROC Curve")
        axes["roc"].legend()

        #Plots the kde plot for Seeing the risk analysis
        
        sns.kdeplot( #This one plots the kde plot for all the 0 where the model has not detected a risk
            self.y_prob[self.y_test == 0],
            fill=True,
            color=palette[1],
            alpha=0.6,
            linewidth=2,
            label="No Failure",
            ax=axes["kde"]
        )

        sns.kdeplot( #This one plots the kde plot for all the 1 where the model has detected a risk
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
        
        #Finds the average precision score which serves as a single-number summary of the Precision-Recall curve
        ap = average_precision_score(self.y_test, self.y_prob)

        axes["prc"].plot(recall, precision, lw=2.5, label=f"AP = {ap:.3f}")
        axes["prc"].set_xlabel("Recall")
        axes["prc"].set_ylabel("Precision")
        axes["prc"].set_title("Precision–Recall Curve")
        axes["prc"].legend()
        
        '''
        Finds where the threshold. [0][-1] here -1 is chose because Picks the highest 
        possible threshold that still satisfies the 90% recall requirement.
        '''
        idx = np.where(recall >= target_recall)[0][-1] 
        
        #Finds which associates as a positive for the model to predict
        risk_tolerance = risk_tolerances[idx] 

        #Filters the y_prob according to the risk_tolerance
        y_pred_thr = (self.y_prob >= risk_tolerance).astype(int)

        cm = confusion_matrix(self.y_test, y_pred_thr)
        tn, fp, fn, tp = cm.ravel()

        cm_percent = cm / cm.sum() * 100
        labels = np.array([
            [f"TN:{tn}\n({cm_percent[0,0]:.1f}%)", f"FP:{fp}\n({cm_percent[0,1]:.1f}%)"],
            [f"FN:{fn}\n({cm_percent[1,0]:.1f}%)", f"TP:{tp}\n({cm_percent[1,1]:.1f}%)"]
        ])

        #Plots the confusion matrix with Induvidual Values and Percentage of TP,FP,TN,FN
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
        
        #Hands everything over to evaluate to print the verbose part for logging
        self.evaluate()
        
    # ─── Verbose Output For The Model ─────────────────────────────────────

    
    def evaluate(self):
        '''
        This function prints the Neccessary data required for evaluating the efficiency of the model
        '''
        tn, fp, fn, tp = confusion_matrix(self.y_test, self.y_pred_thr).ravel() #Gives the FP,FN,TP,TN of the model
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
        
        #Collects the garabage values so that it frees space from the RAM.Best used with extremely large dataset
        gc.collect()

# ─── Failure Classification Model ─────────────────────────────────────────────
'''
This model uses Multinomial Logistic Regression (Softmax) to pinpoint exactly 
which type of failure is occurring by treating all classes within a single 
optimization objective.
'''
class FailureClassificationModel:
    def __init__(self, df: pd.DataFrame, target, classifications, ignore=None):
        self.filename = 'ai4i2020'     # kept for image-save filenames
        self.source_df = df            # already-loaded DataFrame from MySQL
        self.target = target
        self.classifications = classifications  # List of class labels e.g. ['TWF','HDF','PWF','OSF']
        # failure_cols: same ordered list used to build the multi-class target
        self.failure_cols = classifications
        self.ignore = ignore if ignore is not None else []
        
        # Ensures that classifications parameter isn't empty.
        if not self.classifications:
            print('Error classification cannot be empty....')

    # ─── Import The Dataset And Split It ──────────────────────────────────

    def dataset(self):
        df = self.source_df.copy()
        
        # Keep only rows where a machine failure actually occurred
        failure_df = df[df[self.target] == 1].copy()
        
        # Build a multi-class target from individual failure-type columns.
        # Priority order: TWF=0, HDF=1, PWF=2, OSF=3, RNF=4.
        # If multiple failure bits are set, take the first match in self.failure_cols order.
        def _get_class(row):
            for idx, col in enumerate(self.failure_cols):
                if row[col] == 1:
                    return idx
            return len(self.failure_cols) - 1  # fallback: last class
        
        failure_df['_failure_class'] = failure_df.apply(_get_class, axis=1)
        
        y = failure_df['_failure_class'].astype(int)
        
        # Drop all classification-related columns and the binary target from X
        drop_cols = [c for c in (self.ignore + [self.target, '_failure_class']) if c in failure_df.columns]
        X = failure_df.drop(columns=drop_cols)

        # Identify indices of rows without any missing values
        mask = X.notna().all(axis=1)
        
        self.X = X.loc[mask].reset_index(drop=True)
        self.y = y.loc[mask].reset_index(drop=True)

        print(f"Classification dataset: {len(self.X)} failure rows, class distribution:")
        print(self.y.value_counts().sort_index().rename(index={i: self.classifications[i] for i in range(len(self.classifications))}))

        # Splits the data into training and testing data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=0.2, 
            stratify=self.y,
            random_state=42
        )

    # ─── Train And Run A Multinomial Trial ────────────────────────────────

    def train_run(self):
        self.dataset()
        '''
        A pipeline is created so that both the scaler and model will be saved
        '''
        pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("clf", LogisticRegression(           
                solver='saga',
                class_weight="balanced",
                max_iter=5000,
                random_state=42
            ))
        ])

        # Search for optimal regularization strength
        param_dist = {
            "clf__C": np.logspace(-2, 2, 20)
        }

        #RandomizedSearchCV to find the best parameter for LogisticRegression 
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=20,
            scoring="f1_macro", # Balanced metric for multi-class
            cv=3,
            random_state=42,
            n_jobs=-1
        )

        search.fit(self.X_train, self.y_train)
        
        # Generate probabilities and predictions
        self.y_prob = search.predict_proba(self.X_test)
        self.y_pred = search.predict(self.X_test)
        
        # For plotting, we analyze the 'Aggregated' performance
        self.plot()

        model_name = f"models/Risk_eval_Classification.pkl"
        jb.dump(search, model_name)
        
        return model_name

    # ─── Plotting All The Graphs ──────────────────────────────────────────

    def plot(self):
        '''
        This method generates the 4-quadrant dashboard adapted for 
        Multinomial (Softmax) results.
        '''
        palette = sns.color_palette("crest", 10)
        layout = [["confusion", "roc"],
                ["prc", "kde"]]
        
        fig, axes = plt.subplot_mosaic(layout, figsize=(12, 10))

        # ─── Macro-Averaged ROC Curve ─────────────────────────────────
        
        # Calculate One-vs-Rest ROC for the multiclass output
        n_classes = self.y_prob.shape[1]
        for i in range(n_classes):
            label = self.classifications[i] if i < len(self.classifications) else f"Class {i}"
            fpr, tpr, _ = roc_curve(self.y_test == i, self.y_prob[:, i])
            axes["roc"].plot(fpr, tpr, lw=1.5, alpha=0.5, label=f"Class {label}")
        
        axes["roc"].plot([0,1], [0,1], ls="--", color="gray", alpha=0.6)
        axes["roc"].set_xlabel("False Positive Rate")
        axes["roc"].set_ylabel("True Positive Rate")
        axes["roc"].set_title("Multiclass ROC (OvR)")
        axes["roc"].legend(fontsize='small', ncol=2)

        # ─── Probability Distribution (Winning Class) ─────────────────
        
        # Grabs the max probability assigned to the predicted class
        max_probs = np.max(self.y_prob, axis=1)

        sns.kdeplot(
            max_probs,
            fill=True,
            color=palette[5],
            alpha=0.6,
            linewidth=2,
            ax=axes["kde"]
        )
        axes["kde"].set_xlabel("Confidence (Max Probability)")
        axes["kde"].set_title("Softmax Prediction Confidence")

        # ─── Precision Recall Curve (Macro) ───────────────────────────

        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(self.y_test == i, self.y_prob[:, i])
            axes["prc"].plot(recall, precision, lw=1.5, alpha=0.5)

        axes["prc"].set_xlabel("Recall")
        axes["prc"].set_ylabel("Precision")
        axes["prc"].set_title("Multiclass PR Curves")

        # ─── Per-Class Precision / Recall / F1 Bar Chart ─────────────
        # Much more informative than a raw confusion matrix for 4-class softmax.
        # Shows at a glance which failure modes are hardest to distinguish.
        
        from sklearn.metrics import precision_recall_fscore_support
        prec_vals, rec_vals, f1_vals, _ = precision_recall_fscore_support(
            self.y_test, self.y_pred, labels=list(range(n_classes)), zero_division=0
        )
        
        class_labels = [self.classifications[i] if i < len(self.classifications) else f"Class {i}"
                        for i in range(n_classes)]
        x = np.arange(len(class_labels))
        bar_w = 0.25
        bar_palette = sns.color_palette("crest", 3)
        
        axes["confusion"].bar(x - bar_w, prec_vals, width=bar_w, label="Precision",
                              color=bar_palette[0], alpha=0.85)
        axes["confusion"].bar(x,          rec_vals,  width=bar_w, label="Recall",
                              color=bar_palette[1], alpha=0.85)
        axes["confusion"].bar(x + bar_w,  f1_vals,   width=bar_w, label="F1-Score",
                              color=bar_palette[2], alpha=0.85)
        
        axes["confusion"].set_xticks(x)
        axes["confusion"].set_xticklabels(class_labels)
        axes["confusion"].set_ylim(0, 1.05)
        axes["confusion"].set_title("Per-Class Precision / Recall / F1")
        axes["confusion"].set_xlabel("Failure Mode")
        axes["confusion"].set_ylabel("Score")
        axes["confusion"].legend(fontsize='small')
        axes["confusion"].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        fig.savefig(f"images/Dashboard_Classifications_{self.filename}.png", dpi=300)
        plt.show()
        
        self.evaluate()

    # ─── Verbose Output Of The Model ──────────────────────────────────────

    def evaluate(self):
        print("========== Softmax Model Features =========")
        print(*self.X.columns, sep='\n')
        print("\n=== Model Evaluation (Test Set) ===")
        print(classification_report(self.y_test, self.y_pred))
        print("====================================")
        gc.collect()
 
# ─── Complete System ──────────────────────────────────────────────────────────

'''
The below class encorporates both models first the RiskEvalModel runs if the 
model detects a risk FailureClassificationModel kicks in to find what type of 
failure occured.
'''
    
class FailurePredictionSystem:
    def __init__(self, df: pd.DataFrame, target, risk_tolerance, ignore=None, classifications=None,
                 warning_sensitivity=0.6, diagnosis_sensitivity=0.4, persistence_threshold=3):
        
        # Initialisation of variables
        
        self.df = df               # full training DataFrame loaded from MySQL
        self.target = target
        self.risk_tolerance = risk_tolerance
        self.classifications = classifications
        self.ignore = ignore if ignore else []
        self.history = []
        self.risk_model = None
        self.classification_model = None
        self.warning_streak = 0
        self.warning_sensitivity = warning_sensitivity
        self.diagnosis_sensitivity = diagnosis_sensitivity
        self.persistence_threshold = persistence_threshold

    # ─── Load All The Models ──────────────────────────────────────────────

    def LoadModels(self):
        '''
        This function first check if the models exists
        If it exsists then it 
        '''
        try: 
            os.mkdir('models')
        except:
            pass
        risk_model_path = "models/Risk_eval_model.pkl"
        if os.path.exists(risk_model_path): #Checks if an already trained RiskEvalModel exsists
            self.risk_model = jb.load(risk_model_path)
            print("Loaded RiskEvalModel from disk.")
        else: # If RiskEvalModel is not already trained it will trained saved and loaded
            risk_model = RiskEvalModel(
                self.df,
                self.target,
                self.risk_tolerance,
                self.ignore + self.classifications
            )
            risk_model_path = risk_model.train_run()
            self.risk_model = jb.load(risk_model_path)

        '''
        Recursively checks if each of the model for each type of ailure that exsists
        if a model doesnt exists a model is trained, saved and loaded for use.
        '''
        softmax_path = "models/Risk_eval_Classification.pkl"
        if os.path.exists(softmax_path):
            self.classification_model = jb.load(softmax_path) # Singular
        else:
            clf = FailureClassificationModel(
                self.df,
                self.target,
                self.classifications, # Ordered class labels: TWF, HDF, PWF, OSF
                self.ignore
            )
            model_path = clf.train_run()
            self.classification_model = jb.load(model_path)
        
        print('Loaded all FailureClassification Models from disk')
    # ─── Predict The Values ───────────────────────────────────────────────

    '''
    This is the main function that is used to predict data.
    The funtion first imports a row as input to be predicted.
    It creates the features automatically which are required
    by the model to predict the data.
    
    These Features include:
    
        1)The Rolling Mean is a smoothing technique used to identify the underlying trend 
        of a dataset by filtering noise.

        2)Volatility(Rolling Standard deviation) measures the dispersion of data points around 
        the mean over a specific window. In predictive modeling, it is the primary indicator of
        risk or instability.
    
        3)Delta represents the absolute change between the current value and a previous value. 
        It shifts the focus from the "level" of the data to the "change" in the data.
        
        4)The Rolling Delta is a second-order feature. It typically measures the average change 
        (the average Delta) over a specific window, or the difference between two rolling means.
    '''
    def predict(self, X: pd.DataFrame):
        '''
        This is the main function that is used to predict data.
        Updated to support a single Softmax classification model for diagnosis.
        '''
        
        # When multiple rows of data is present, this piece code filters out the most recent one.
        current_row_dict = X.iloc[0].to_dict()
        self.history.append(current_row_dict)
        if len(self.history) > 5:
            self.history.pop(0)

        current_data_df = X.copy() # Creating a copy to preserve original data
        history_dataframe = pd.DataFrame(self.history)
        
        cols_to_exclude = [self.target] + self.classifications + self.ignore
        
        dynamic_features = [ # includes only the data necessary for the model
            c for c in X.columns 
            if c not in cols_to_exclude 
            and not any(x in c for x in ['_Rolling_Mean', '_Volatility', '_Delta', '_Rolling_Delta'])
        ]

        for col in dynamic_features:
            current_data_df[f'{col}_Rolling_Mean'] = history_dataframe[col].mean() 
            
            if len(self.history) > 1: # so that it doesn't throw a ZeroError
                current_data_df[f'{col}_Volatility'] = history_dataframe[col].std()
                current_data_df[f'{col}_Delta'] = self.history[-1][col] - self.history[-2][col]
                current_data_df[f'{col}_Rolling_Delta'] = history_dataframe[col].diff().mean()
            else:
                current_data_df[f'{col}_Volatility'] = 0.0
                current_data_df[f'{col}_Delta'] = 0.0
                current_data_df[f'{col}_Rolling_Delta'] = 0.0

        # Ensures no NaN values creep in
        current_data_df = current_data_df.fillna(0)

        # Cleans up the data one last time and puts the features in the exact order the model expects.
        inference_df = current_data_df.drop(columns=[c for c in cols_to_exclude if c in current_data_df.columns])
        inference_df = inference_df[self.risk_model.feature_names_in_]
        
        # ─── Level 1: Risk Probability ─────────────────────────────────────
        # Gives the exact probability of a Failure happening
        raw_risk_prob = float(self.risk_model.predict_proba(inference_df)[:, 1][0])
        
        # creates a list called prob_history
        if not hasattr(self, 'prob_history'):
            self.prob_history = []
        
        # Saves the last 3 predictions    
        self.prob_history.append(raw_risk_prob)
        if len(self.prob_history) > 3: 
            self.prob_history.pop(0)

        risk_prob = sum(self.prob_history) / len(self.prob_history)
        
        warning_zone = self.risk_tolerance * self.warning_sensitivity
        
        # Update Streak Counters
        if risk_prob >= warning_zone: 
            self.warning_streak += 1
        else:
            self.warning_streak = 0

        # Determine Alert Status with INTENSITY OVERRIDE
        if risk_prob >= self.risk_tolerance:
            alert_status = "CRITICAL"
            
        elif risk_prob >= 0.60: 
            alert_status = "WARNING" 
            # Force streak to max to keep alert active if risk drops slightly
            self.warning_streak = max(self.warning_streak, self.persistence_threshold)
            
        elif risk_prob >= warning_zone:
            if self.warning_streak >= self.persistence_threshold:
                alert_status = "WARNING"
            else:
                alert_status = "HEALTHY"
        else:
            alert_status = "HEALTHY"

        # ─── Level 2: Softmax Diagnosis ────────────────────────────────────
        # Run the single multinomial model pass
        softmax_probs = self.classification_model.predict_proba(inference_df)[0]
        
        # Map probabilities back to the specific classes the model was trained on
        model_classes = self.classification_model.classes_
        results = {label: 0.0 for label in self.classifications}
        
        for i, prob in enumerate(softmax_probs):
            class_idx = model_classes[i]
            # Map index back to string label if classifications list is provided
            label = self.classifications[class_idx]
            results[label] = round(float(prob), 4)
        
        active = [f for f, p in results.items() if p >= self.diagnosis_sensitivity]
        failure = max(results, key=results.get) if active else "RandomFailure"

        return {
            "risk_prob": round(risk_prob, 4), 
            "alert_level": alert_status, 
            "failure_type": failure if alert_status != "HEALTHY" else None,
            "streak": self.warning_streak,
            "details": results
        }
    # ─── Finds The Best Values For Warnings To Trigger ────────────────────

    '''
    This function is used to calibrate the warning and diagnosis sensitivity
    values of the alert system using a brute-force search.

    The main objective of this calibration is to find the combination that
    produces the first non-HEALTHY alert as close as possible to the
    expected warning step.

    The calibration process includes:

        1)Trying multiple warning sensitivity values and diagnosis sensitivity
          values using a brute-force grid.

        2)Simulating streaming behaviour by predicting one row at a time
          and updating the internal history and streak counters.

        3)Recording the first time step at which the system raises any
          alert (i.e., when the alert level is no longer HEALTHY).

        4)Scoring each parameter combination based on how close the first
          alert step is to the target warning step.

        5)Selecting and returning the sensitivity values that achieve the
          best score.
    '''
    def calibrate_sensitivities(self, test_df, target_step_for_warning=4):
            best_score = -1
            best_params = {}
            
            # Total no of combinations is 20
            warning_range = np.linspace(0.1,1, 10) # 5 values for warning
            diagnosis_range = np.linspace(0.1, 1, 10) # 4 values for diagnosis
            
            '''
            Uncomment if you wanna change the values
            '''
            #print(f"{'Warning_S':<10} | {'Diag_S':<10} | {'Warning Step':<12} | {'Score'}")
            #print("-" * 50)
            
            '''
            Trying to Brute Force every single of the 20 combinations to fins the best one
            '''
            for w_s in warning_range:
                for d_s in diagnosis_range:
                    self.warning_sensitivity = w_s
                    self.diagnosis_sensitivity = d_s
                    self.warning_streak = 0
                    self.history = []
                    
                    first_warning_step = None
                    
                    for i in range(len(test_df)):
                        # Predictions are first made using the the current combination
                        res = self.predict(test_df.iloc[[i]])
                        
                        # Finds the first "Non Healthy" row 
                        if res['alert_level'] != "HEALTHY" and first_warning_step is None:
                            first_warning_step = i + 1
                    
                    # Checks how close the prediction was to the actual value
                    if first_warning_step:
                        score = 1.0 / (abs(target_step_for_warning - first_warning_step) + 1)
                    else:
                        score = 0
                    
                    '''
                    Uncomment the below statement if you want to change the values
                    '''
                    #print(f"{w_s:<10.2f} | {d_s:<10.2f} | {str(first_warning_step):<12} | {score:.2f}")
                    
                    # Stores the best params
                    if score > best_score:
                        best_score = score
                        best_params = {'warning_sensitivity': w_s, 'diagnosis_sensitivity': d_s}
                        
                    if score == 1.0:
                        break
            
            print("-" * 50)
            print(f"Optimal Parameters: {best_params}")
            print("-" * 50)
            # Resets everything so it does not effect actual predictions
            self.history = []
            if hasattr(self, 'prob_history'):
                self.prob_history = [] 
            self.warning_streak = 0
            
            return best_params

    def evaluate_test_set(self, test_df):
        '''
        Helper function to run the prediction loop for simulation
        '''
        print(f"\n{'Step':<5} | {'Risk':<8} | {'Alert':<10} | {'Failure Type'}")
        print("-" * 50)
        for i in range(len(test_df)):
            result = self.predict(test_df.iloc[[i]])
            print(f"{i+1:<5} | {result['risk_prob']:<8.4f} | {result['alert_level']:<10} | {result['failure_type']}")
        
if __name__ == '__main__':
    target_labels = ['TWF','HDF','PWF','OSF']
    # RNF is intentionally excluded from classification — it is the unclassifiable fallback
    ignore_list = ['RNF','UDI','Product ID','Type'] + target_labels

    # Load training data from MySQL instead of CSV
    training_df = load_dataframe()

    FinalSystem = FailurePredictionSystem(
        training_df,
        'Machine failure',
        0.765, # After a lot of trial and error this is sort of the best risk_tolerance i found
        ignore_list,
        target_labels
    )
    FinalSystem.LoadModels()

    # Fake data based on the real dataset so that by the time the last row is
    # reached a HDF Failure will only happen
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
