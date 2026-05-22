"""
Step 3: Hyperparameter Tuning with Hyperopt
=============================================
Uses Hyperopt's Bayesian optimization to find the best Random Forest
hyperparameters. Each trial is logged to MLflow.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# -----------------------------------------------------------------
# Step 1: Load the prepared data
# -----------------------------------------------------------------
print("Loading data...")
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

X_train = train_df.drop(columns=["RainTomorrow"])
y_train = train_df["RainTomorrow"]
X_test  = test_df.drop(columns=["RainTomorrow"])
y_test  = test_df["RainTomorrow"]

print(f"   Train: {len(X_train):,} | Test: {len(X_test):,}")

# -----------------------------------------------------------------
# Step 2: Set the MLflow experiment
# -----------------------------------------------------------------
# Use the SAME experiment as before so we can compare baseline vs tuned.
mlflow.set_experiment("rain-prediction")

# -----------------------------------------------------------------
# Step 3: Define the search space
# -----------------------------------------------------------------
# Hyperopt will sample values from these ranges for each trial.
# - hp.quniform = uniform integer values
# - hp.uniform  = uniform float values
search_space = {
    "n_estimators":     hp.quniform("n_estimators", 50, 200, 10),
    "max_depth":        hp.quniform("max_depth", 5, 25, 1),
    "min_samples_split": hp.quniform("min_samples_split", 2, 30, 1),
    "min_samples_leaf":  hp.quniform("min_samples_leaf", 1, 10, 1),
}

# -----------------------------------------------------------------
# Step 4: Define the objective function
# -----------------------------------------------------------------
# Hyperopt calls this function once per trial.
# It must return a "loss" - Hyperopt tries to MINIMIZE this.
# Since we want to MAXIMIZE F1 score, we return NEGATIVE F1 as loss.
def objective(params):
    # Hyperopt gives us floats - convert integer params back to int
    params["n_estimators"]      = int(params["n_estimators"])
    params["max_depth"]         = int(params["max_depth"])
    params["min_samples_split"] = int(params["min_samples_split"])
    params["min_samples_leaf"]  = int(params["min_samples_leaf"])

    # Start a NESTED MLflow run for this trial
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForest-Tuned")

        model = RandomForestClassifier(
            **params,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        print(f"   Trial f1={metrics['f1']:.4f} | "
              f"n_est={params['n_estimators']} depth={params['max_depth']}")

        # Hyperopt minimizes loss, so we negate the F1
        return {"loss": -metrics["f1"], "status": STATUS_OK}

# -----------------------------------------------------------------
# Step 5: Run the hyperparameter search
# -----------------------------------------------------------------
NUM_TRIALS = 15   # Number of different combinations to try

print(f"\nStarting hyperparameter search ({NUM_TRIALS} trials)...")
print("(This will take 3-7 minutes)\n")

with mlflow.start_run(run_name="hyperopt-tuning-parent"):
    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,   # TPE = Tree-structured Parzen Estimator (smart Bayesian search)
        max_evals=NUM_TRIALS,
        trials=trials,
        rstate=np.random.default_rng(42),
    )

    # Convert floats back to ints for the final best params
    best_params["n_estimators"]      = int(best_params["n_estimators"])
    best_params["max_depth"]         = int(best_params["max_depth"])
    best_params["min_samples_split"] = int(best_params["min_samples_split"])
    best_params["min_samples_leaf"]  = int(best_params["min_samples_leaf"])

    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    best_f1 = -min(trials.losses())
    mlflow.log_metric("best_f1", best_f1)

print("\n" + "=" * 60)
print("Hyperparameter tuning complete!")
print("=" * 60)
print(f"Best F1 score: {best_f1:.4f}")
print(f"Best parameters: {best_params}")
print("\nOpen MLflow UI to compare all tuning trials.")