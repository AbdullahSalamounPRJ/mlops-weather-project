"""
Step 6: Performance Monitoring & Drift Simulation
==================================================
Simulates new data coming in over several "weeks" and monitors how
the deployed model's performance changes. Logs everything to MLflow
and produces a drift visualization.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# -----------------------------------------------------------------
# Step 1: Load the registered production model
# -----------------------------------------------------------------
MODEL_NAME = "rain-classifier"
MODEL_URI  = f"models:/{MODEL_NAME}@production"

print(f"Loading model from registry: {MODEL_URI}")
model = mlflow.sklearn.load_model(MODEL_URI)
print("   Model loaded successfully!")

# -----------------------------------------------------------------
# Step 2: Load the test data (simulates incoming production data)
# -----------------------------------------------------------------
test_df = pd.read_csv("test.csv")
X_test = test_df.drop(columns=["RainTomorrow"])
y_test = test_df["RainTomorrow"]

print(f"   Total observations available: {len(test_df):,}")

# -----------------------------------------------------------------
# Step 3: Split data into 8 "weeks" of production data
# -----------------------------------------------------------------
NUM_WEEKS = 8
batch_size = len(test_df) // NUM_WEEKS
weekly_metrics = []

# -----------------------------------------------------------------
# Step 4: Set up MLflow experiment for monitoring
# -----------------------------------------------------------------
mlflow.set_experiment("rain-prediction-monitoring")

print(f"\nSimulating {NUM_WEEKS} weeks of production data...\n")

with mlflow.start_run(run_name="drift-monitoring-run"):
    np.random.seed(42)  # so the simulated drift is reproducible

    for week in range(NUM_WEEKS):
        start = week * batch_size
        end   = start + batch_size

        X_week = X_test.iloc[start:end].copy()
        y_week = y_test.iloc[start:end]

        # ----------------------------------------------------
        # Simulate concept drift starting at week 6
        # (in real life this could be caused by climate change,
        # broken sensors, or shifting weather patterns)
        # ----------------------------------------------------
        if week >= 5:
            X_week["MinTemp"]     = X_week["MinTemp"]     + np.random.normal(4, 1, len(X_week))
            X_week["MaxTemp"]     = X_week["MaxTemp"]     + np.random.normal(4, 1, len(X_week))
            X_week["Humidity9am"] = X_week["Humidity9am"] - np.random.uniform(15, 25, len(X_week))
            X_week["Humidity3pm"] = X_week["Humidity3pm"] - np.random.uniform(15, 25, len(X_week))

        # Make predictions
        y_pred = model.predict(X_week)

        # Calculate metrics
        metrics = {
            "accuracy":  accuracy_score(y_week, y_pred),
            "f1":        f1_score(y_week, y_pred),
            "precision": precision_score(y_week, y_pred, zero_division=0),
            "recall":    recall_score(y_week, y_pred),
        }

        # Log each week's metrics with a "step" so MLflow plots them as a time series
        for name, value in metrics.items():
            mlflow.log_metric(name, value, step=week)

        weekly_metrics.append({"week": week + 1, **metrics})

        drift_flag = " [DRIFT]" if week >= 5 else ""
        print(f"   Week {week+1}: accuracy={metrics['accuracy']:.4f}, "
              f"f1={metrics['f1']:.4f}, recall={metrics['recall']:.4f}{drift_flag}")

    # -----------------------------------------------------------------
    # Step 5: Create the drift visualization
    # -----------------------------------------------------------------
    print("\nCreating drift visualization...")
    df_monitor = pd.DataFrame(weekly_metrics)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_monitor["week"], df_monitor["accuracy"], marker="o", label="Accuracy",  linewidth=2)
    ax.plot(df_monitor["week"], df_monitor["f1"],       marker="s", label="F1 Score",  linewidth=2)
    ax.plot(df_monitor["week"], df_monitor["recall"],   marker="^", label="Recall",    linewidth=2)
    ax.axvline(x=5.5, color="red", linestyle="--", alpha=0.6, label="Drift starts here")

    ax.set_xlabel("Week", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title("Model Performance Over Time (Drift Monitoring)", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plot_path = "drift_monitoring.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    plt.close()

    # Log the chart as an artifact
    mlflow.log_artifact(plot_path)

    # Save the weekly metrics as a CSV and log it too
    csv_path = "weekly_metrics.csv"
    df_monitor.to_csv(csv_path, index=False)
    mlflow.log_artifact(csv_path)

    print(f"   Saved plot: {plot_path}")
    print(f"   Saved CSV:  {csv_path}")

# -----------------------------------------------------------------
# Step 6: Summary report
# -----------------------------------------------------------------
print("\n" + "=" * 60)
print("Monitoring complete!")
print("=" * 60)

avg_before = df_monitor[df_monitor["week"] <= 5]["accuracy"].mean()
avg_after  = df_monitor[df_monitor["week"] >  5]["accuracy"].mean()
drift_drop = avg_before - avg_after

print(f"Average accuracy BEFORE drift (weeks 1-5): {avg_before:.4f}")
print(f"Average accuracy AFTER drift  (weeks 6-8): {avg_after:.4f}")
print(f"Performance drop:                          {drift_drop:.4f} ({drift_drop*100:.1f}%)")

if drift_drop > 0.02:
    print("\nALERT: Significant drift detected! Model needs retraining.")
else:
    print("\nModel performance is stable.")

print(f"\nView the chart at: drift_monitoring.png")
print(f"View this run in the MLflow UI under experiment 'rain-prediction-monitoring'")