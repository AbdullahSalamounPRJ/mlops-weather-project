"""
Step 4: Model Registry
=======================
Trains the FINAL model using the best parameters found by Hyperopt,
saves it with MLflow, registers it in the Model Registry, and promotes
it through stages (None -> Staging -> Production) using aliases.
"""

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from mlflow.tracking import MlflowClient

# -----------------------------------------------------------------
# Step 1: Best parameters discovered by Hyperopt (from tune.py)
# -----------------------------------------------------------------
BEST_PARAMS = {
    "max_depth":         19,
    "min_samples_leaf":  5,
    "min_samples_split": 7,
    "n_estimators":      160,
    "random_state":      42,
    "n_jobs":            -1,
}

MODEL_NAME = "rain-classifier"

# -----------------------------------------------------------------
# Step 2: Load the data
# -----------------------------------------------------------------
print("Loading data...")
train_df = pd.read_csv("train.csv")
test_df  = pd.read_csv("test.csv")

X_train = train_df.drop(columns=["RainTomorrow"])
y_train = train_df["RainTomorrow"]
X_test  = test_df.drop(columns=["RainTomorrow"])
y_test  = test_df["RainTomorrow"]

# -----------------------------------------------------------------
# Step 3: Train the FINAL model with best params and register it
# -----------------------------------------------------------------
mlflow.set_experiment("rain-prediction")

print("\nTraining final model with best parameters...")
with mlflow.start_run(run_name="final-tuned-model") as run:
    mlflow.log_params(BEST_PARAMS)
    mlflow.log_param("model_type", "RandomForest-Final")
    mlflow.set_tag("stage", "production-candidate")

    model = RandomForestClassifier(**BEST_PARAMS)
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

    # Log and REGISTER the model in one call
    # The registered_model_name parameter is the key part - it adds
    # the model to the Registry automatically.
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=MODEL_NAME,
    )

    run_id = run.info.run_id
    print(f"   Run ID: {run_id}")
    for name, value in metrics.items():
        print(f"   {name}: {value:.4f}")

# -----------------------------------------------------------------
# Step 4: Promote the registered model through stages
# -----------------------------------------------------------------
# Modern MLflow uses "aliases" instead of the old stage strings.
# We will assign two aliases: "staging" and "production".
print("\nPromoting model through stages...")

client = MlflowClient()

# Get the latest version of the registered model
versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(int(v.version) for v in versions)
print(f"   Latest version of '{MODEL_NAME}': {latest_version}")

# Assign 'staging' alias first
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="staging",
    version=latest_version,
)
print(f"   Assigned alias 'staging' to version {latest_version}")

# Then promote to 'production'
client.set_registered_model_alias(
    name=MODEL_NAME,
    alias="production",
    version=latest_version,
)
print(f"   Assigned alias 'production' to version {latest_version}")

# Add a description so users know what this model does
client.update_registered_model(
    name=MODEL_NAME,
    description="Predicts whether it will rain tomorrow in Australia. "
                "Trained on weatherAUS.csv. Random Forest tuned with Hyperopt.",
)

print("\n" + "=" * 60)
print("Model successfully registered and promoted to Production!")
print("=" * 60)
print(f"Model name: {MODEL_NAME}")
print(f"Version:    {latest_version}")
print(f"Aliases:    staging, production")
print(f"\nCheck the 'Model registry' tab in the MLflow UI to see it.")