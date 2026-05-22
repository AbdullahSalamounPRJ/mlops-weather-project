"""
Step 2: Train Multiple Models with MLflow Tracking
===================================================
Trains 3 different models on the prepared data and logs every
experiment to MLflow so we can compare them in the dashboard.
"""

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------
# Step 1: Load the prepared data
# -----------------------------------------------------------------
print("Loading prepared data...")
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

# Separate features (X) and target (y)
X_train = train_df.drop(columns=["RainTomorrow"])
y_train = train_df["RainTomorrow"]
X_test  = test_df.drop(columns=["RainTomorrow"])
y_test  = test_df["RainTomorrow"]

print(f"   Train samples: {len(X_train):,}")
print(f"   Test samples:  {len(X_test):,}")

# -----------------------------------------------------------------
# Step 2: Scale the features
# -----------------------------------------------------------------
# Many models (like Logistic Regression) work better when all
# features have similar ranges. StandardScaler makes each column
# have mean=0 and std=1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# -----------------------------------------------------------------
# Step 3: Set up the MLflow experiment
# -----------------------------------------------------------------
# All runs from this script will be grouped under this experiment name.
mlflow.set_experiment("rain-prediction")

# -----------------------------------------------------------------
# Step 4: Define a helper function to compute metrics
# -----------------------------------------------------------------
def evaluate(y_true, y_pred, y_proba):
    """Calculate all our evaluation metrics."""
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall":    recall_score(y_true, y_pred),
        "roc_auc":   roc_auc_score(y_true, y_proba),
    }

# -----------------------------------------------------------------
# Step 5: Train Model 1 - Logistic Regression
# -----------------------------------------------------------------
print("\n--- Model 1: Logistic Regression ---")
with mlflow.start_run(run_name="logistic-regression"):
    params = {
        "C": 1.0,
        "max_iter": 1000,
        "solver": "liblinear",
        "random_state": 42,
    }
    mlflow.log_params(params)
    mlflow.log_param("model_type", "LogisticRegression")

    model = LogisticRegression(**params)
    model.fit(X_train_scaled, y_train)

    y_pred  = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = evaluate(y_test, y_pred, y_proba)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")

    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")

# -----------------------------------------------------------------
# Step 6: Train Model 2 - Decision Tree
# -----------------------------------------------------------------
print("\n--- Model 2: Decision Tree ---")
with mlflow.start_run(run_name="decision-tree"):
    params = {
        "max_depth": 10,
        "min_samples_split": 20,
        "random_state": 42,
    }
    mlflow.log_params(params)
    mlflow.log_param("model_type", "DecisionTree")

    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)   # Trees don't need scaling

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_pred, y_proba)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")

    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")

# -----------------------------------------------------------------
# Step 7: Train Model 3 - Random Forest
# -----------------------------------------------------------------
print("\n--- Model 3: Random Forest ---")
with mlflow.start_run(run_name="random-forest"):
    params = {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 10,
        "random_state": 42,
        "n_jobs": -1,  # Use all CPU cores
    }
    mlflow.log_params(params)
    mlflow.log_param("model_type", "RandomForest")

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_pred, y_proba)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")

    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")

print("\nDone! All 3 models trained and logged to MLflow.")
print("Now run 'mlflow ui' to compare them in the dashboard.")