from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score, confusion_matrix , ConfusionMatrixDisplay
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

sns.set_theme(
    context="talk",   
    palette="crest_r"
)

# ─── Sets Up Automatic Logging ────────────────────────────────────────────────

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
        filename_modified, ext = os.path.splitext(filename) #splits the entire filename into the filename and extension
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
    
    # ─── Import The Dataset And Split The Data In Two ─────────────────────
    
    def dataset(self, filename: str,target: str,failure_only = False,ignore=None):
        '''
        This function reads the dataset and splits the dataset into training and testing data
        '''
        if ignore is None:
            ignore = []

        self.df = pd.read_csv(filename)

        if failure_only:
            self.df = df[df[target] == 1]
        
        y = self.df[target].astype(int)
        X = self.df.drop(columns=ignore + [target])

        mask = X.notna().all(axis=1)
        self.X = X.loc[mask].reset_index(drop=True)
        self.y = y.loc[mask].reset_index(drop=True)

        idx = int(0.8 * len(X))

        self.X_train = X.iloc[:idx]
        self.y_train = y.iloc[:idx]

        self.X_test  = X.iloc[idx:]
        self.y_test  = y.iloc[idx:]
        
    # ─── Train The Model And Run A Trial ──────────────────────────────────

    
    def train_run(self):
        '''
        Created a pipline where RobustScaler is first applied and then
        piped to Logistic Regression for model fitting
        '''
        
        self.dataset(self.filename+self.ext,self.target,False,self.ignore)
        
        #Creates the pipline for the model to work
        pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                l1_ratio=1,       # Pure L1
                solver='liblinear', # Supports L1
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
        model_name =  "/models/Risk_eval_model.pkl"
        jb.dump(search,model_name)
        
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
This model uses a number of LogisticRegression to pinpoint what type of failure 
is going to occur. This is achieved by making a model for each type of failure
that may occur.
'''
class FailureClassificationModel:
    def __init__(self,filename,target,risk_tolerance,ignore=None,classifications = None):
        filename_modified, ext = os.path.splitext(filename) #splits the entire filename into the filename and extension
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
        
        #Makes sure that if ignore paramter is not passes it is turned into a list
        if ignore == None:
            self.ignore = []
        else:
            self.ignore = ignore
        
        #Eh dont need these vars
        del ignore
        del filename_modified
        del ext
        
        #Ensures that classifications parmameter isnt empty.
        #Because this pases the types of errors which is used to create a model for each error
        if self.classifications == None or self.classifications == []:
            print('Error classification cannot be empty....')
        
    # ─── Import The Dataset And Split It ──────────────────────────────────

    
    def dataset(self, filename: str,target: str,filter: str,failure_only = False,ignore=None):
        if ignore is None:
            ignore = []

        df = pd.read_csv(filename)

        df = df[df[filter] == 1]
        
        y = df[target].astype(int)
        X = df.drop(columns=ignore + [target])

        # Identify indices of rows without any missing values
        mask = X.notna().all(axis=1)
        
        #Finally saves X and y
        self.X = X.loc[mask].reset_index(drop=True)
        self.y = y.loc[mask].reset_index(drop=True)

        #Splits the data into training and testing data using scikit-learn's built in function
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                                                            self.X, self.y, 
                                                            test_size=0.2, 
                                                            stratify=y,
                                                            random_state=42
                                                        )
        
    # ─── Train And Run A Trial ────────────────────────────────────────────

   
    def train_run(self, param):
        remaining_params = [i for i in self.classifications if i != param]
        self.dataset(self.filename + self.ext,
                    param,
                    self.target,
                    True,
                    self.ignore + remaining_params + [self.target]
                    )
        
        # CHANGED: Added Hyperparameter Tuning to the Classification Step
        pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                l1_ratio=1,       # Pure L1
                solver='liblinear', # Supports L1
                max_iter=5000
            ))
        ])

        # Search for the optimal C to separate the classes
        param_dist = {
            "clf__C": np.logspace(-2, 2, 20) 
        }

        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=20,
            scoring="roc_auc",
            cv=3,  # Lower CV because failure-specific datasets are smaller
            random_state=42,
            n_jobs=-1,
            verbose=0,
            error_score='raise'
        )

        search.fit(self.X_train, self.y_train)
        
        # Predict using the best found model
        self.y_prob = search.predict_proba(self.X_test)[:, 1]
        self.roc = roc_auc_score(self.y_test, self.y_prob)
        
        self.plot(self.roc, 0.50, param)

        model_name = f"/models/Risk_eval_{param}.pkl"
        jb.dump(search, model_name) # Dump the search object, not just the pipe
        
        return model_name
    
    # ─── Plotting All The Graphs ──────────────────────────────────────────

    
    def plot(self,roc,target_recall,param):
        #TODO: CURRENTLY THIS FUNCTION IS REDUNDANT AND THE ALMOST THE SAME AS THE plot()
        #IN RiskEval AND ONLY A MARGINAL DIFFERENCE
        '''
        This method is for making 4 different graphs integrated into a single dashboard
        1) KDE plot of risk
        2) Confusion Matrix
        3) ROC-AUC graph
        4) Precision-Recall Graph
        '''
        palette = sns.color_palette("crest", 10)

        layout = [["confusion", "roc"],
                ["prc", "kde"]]
        
        fig, axes = plt.subplot_mosaic(layout, figsize=(12, 10))

        # ─── ROC-AUC Curve ────────────────────────────────────────────

        #Stores the False positive rate and True Positive rate which is to be plotted
        fpr, tpr, _ = roc_curve(self.y_test, self.y_prob)

        axes["roc"].plot(fpr, tpr, lw=2.5, label=f"AUC = {roc:.3f}")
        axes["roc"].plot([0,1], [0,1], ls="--", color="gray", alpha=0.6)
        axes["roc"].fill_between(fpr, tpr, alpha=0.15)

        axes["roc"].set_xlabel("False Positive Rate")
        axes["roc"].set_ylabel("True Positive Rate")
        axes["roc"].set_title("ROC Curve")
        axes["roc"].legend()

        # ─── Risk Score Distribution ──────────────────────────────────
        
        sns.kdeplot(  #prints the kde plot for no failures
            self.y_prob[self.y_test == 0],
            fill=True,
            color=palette[1],
            alpha=0.6,
            linewidth=2,
            label="No Failure",
            ax=axes["kde"]
        )

        sns.kdeplot(  #prints the kde plot for failures
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

        # ─── Precision Recall Curve ───────────────────────────────────

        precision, recall, risk_tolerances = precision_recall_curve(self.y_test, self.y_prob)
        
        #Finds the average precision score which serves as a single-number summary of the Precision-Recall curve
        ap = average_precision_score(self.y_test, self.y_prob)

        axes["prc"].plot(recall, precision, lw=2.5, label=f"AP = {ap:.3f}")
        axes["prc"].set_xlabel("Recall")
        axes["prc"].set_ylabel("Precision")
        axes["prc"].set_title("Precision–Recall Curve")
        axes["prc"].legend()

        # ─── Confusion Matrix ─────────────────────────────────────────
        '''
        Finds where the threshold. [0][-1] here -1 is chose because Picks the highest 
        possible threshold that still satisfies the 90% recall requirement.
        '''
        idx = np.where(recall >= target_recall)[0][-1]
        
        #Finds which associates as a positive for the model to predict
        self.risk_tolerance = risk_tolerances[idx]
        
        #Filters the y_prob according to the risk_tolerance
        self.y_pred_thr = (self.y_prob >= self.risk_tolerance).astype(int)

        cm = confusion_matrix(self.y_test, self.y_pred_thr)
        tn, fp, fn, tp = cm.ravel()

        cm_percent = cm / cm.sum() * 100 #Array of induvidual percentages of TN,FP,FN,TP
        labels = np.array([ #Creates the induavidual label for each part of the confusion matrix
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
        
        # Automatically Calls Evaluate
        self.evaluate()
    
    # ─── Verbose Output Of The Model ──────────────────────────────────────

    
    def evaluate(self):
        '''
        This function prints the Neccessary data required for evaluating the efficiency of the model
        '''
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
        
        # Initialisation of variables
        
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
        risk_model_path = "/models/Risk_eval_model.pkl"
        if os.path.exists(risk_model_path): #Checks if an already trained RiskEvalModel exsists
            self.risk_model = jb.load(risk_model_path)
            print("Loaded RiskEvalModel from disk.")
        else: # If RiskEvalModel is not already trained it will trained saved and loaded
            risk_model = RiskEvalModel( 
                self.filename,
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
        for failure in self.classifications:
            model_path = f"/models/Risk_eval_{failure}.pkl"
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
        
        # When multiple rows of data is present, this piece code filters out the most recent one.
        current_row_dict = X.iloc[0].to_dict()
        self.history.append(current_row_dict)
        if len(self.history) > 5:
            self.history.pop(0)

        current_data_df = X.copy() #Creating a copy to preserve original data
        history_dataframe = pd.DataFrame(self.history)
        
        cols_to_exclude = [self.target] + self.classifications + self.ignore
        
        dynamic_features = [ #includes only the data necessary for the model
            c for c in X.columns 
            if c not in cols_to_exclude 
            and not any(x in c for x in ['_Rolling_Mean', '_Volatility', '_Delta', '_Rolling_Delta'])
        ]

        for col in dynamic_features:
            #Rolling mean doest need the Safety net
            current_data_df[f'{col}_Rolling_Mean'] = history_dataframe[col].mean() 
            
            if len(self.history) > 1: #acts a safety net so that it does throw a ZeroError
                current_data_df[f'{col}_Volatility'] = history_dataframe[col].std()
                current_data_df[f'{col}_Delta'] = self.history[-1][col] - self.history[-2][col]
                current_data_df[f'{col}_Rolling_Delta'] = history_dataframe[col].diff().mean()
            else:
                current_data_df[f'{col}_Volatility'] = 0.0
                current_data_df[f'{col}_Delta'] = 0.0
                current_data_df[f'{col}_Rolling_Delta'] = 0.0

        # Ensures no NaN values creep in
        current_data_df = current_data_df.fillna(0)

        # Cleans up the data one ladst time and puts the features in the exact order the model expects.
        inference_df = current_data_df.drop(columns=[c for c in cols_to_exclude if c in current_data_df.columns])
        inference_df = inference_df[self.risk_model.feature_names_in_]
        
        # Gives the exact probabilty of a Failure happening
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
            # Optional: Force streak to max to keep alert active if risk drops slightly
            self.warning_streak = max(self.warning_streak, self.persistence_threshold)
            
        elif risk_prob >= warning_zone:
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
        
    def evaluate_test_set(self, test_dataframe_X): #Just a method to give output during testing
        print(f"{'Step':<5} | {'Risk':<7} | {'Streak':<6} | {'Alert':<10} | {'Diagnosis'}")
        print("-" * 65)
        for i in range(len(test_dataframe_X)):
            res = self.predict(test_dataframe_X.iloc[[i]])
            print(f"{i+1:<5} | {res['risk_prob']:<7.4f} | {res['streak']:<6} | {res['alert_level']:<10} | {res['failure_type']}")
    
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
            warning_range = np.linspace(0.3, 0.7, 5) # 5 values for warning
            diagnosis_range = np.linspace(0.2, 0.5, 4) # 4 values for diagnosis
            
            print(f"{'Warning_S':<10} | {'Diag_S':<10} | {'Warning Step':<12} | {'Score'}")
            print("-" * 50)
            
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
                    
                    print(f"{w_s:<10.2f} | {d_s:<10.2f} | {str(first_warning_step):<12} | {score:.2f}")
                    
                    # Stores the best params
                    if score > best_score:
                        best_score = score
                        best_params = {'warning_sensitivity': w_s, 'diagnosis_sensitivity': d_s}
            
            print("-" * 50)
            print(f"Optimal Parameters: {best_params}")
            
            # Resets everything so it does not effect actual predictions
            self.history = []
            if hasattr(self, 'prob_history'):
                self.prob_history = [] 
            self.warning_streak = 0
            
            return best_params
        
if __name__ == '__main__':
    target_labels = ['TWF','HDF','PWF','OSF']
    ignore_list = ['RNF','UDI','Product ID','Type'] + target_labels
    
    FinalSystem = FailurePredictionSystem(
        'ai4i2020.csv',
        'Machine failure',
        0.765, # After a lot of trial and error this is sort of the best risk_tolerance i found
        ignore_list,
        target_labels
    )
    FinalSystem.LoadModels()

    #Fake data based on the real dataset so that by the time the last row is 
    #reached a HDF Failure will only happen
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