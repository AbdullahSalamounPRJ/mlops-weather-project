"""
Step 5: Prediction Client
=========================
Sends sample weather data to the deployed model via HTTP and prints
the predictions. This simulates a real application using the model.

Run this AFTER starting:
    mlflow models serve -m "models:/rain-classifier@production" -p 5001 --no-conda
"""

import pandas as pd
import requests
import json

# -----------------------------------------------------------------
# Step 1: The URL of our deployed model
# -----------------------------------------------------------------
# MLflow always exposes the /invocations endpoint for predictions.
URL = "http://127.0.0.1:5001/invocations"

# -----------------------------------------------------------------
# Step 2: Get some real test data to send
# -----------------------------------------------------------------
# We'll grab the first 5 rows from test.csv as our sample input.
test_df = pd.read_csv("test.csv")

# Pick 5 RANDOM weather observations to predict
random_sample = test_df.sample(n=5, random_state=None).reset_index(drop=True)
X_sample = random_sample.drop(columns=["RainTomorrow"])
y_actual = random_sample["RainTomorrow"]

print("Sample input (5 weather observations):")
print(X_sample)

# -----------------------------------------------------------------
# Step 3: Format the data for MLflow's REST API
# -----------------------------------------------------------------
# MLflow expects a specific JSON format with a "dataframe_split" key.
payload = {
    "dataframe_split": {
        "columns": list(X_sample.columns),
        "data":    X_sample.values.tolist(),
    }
}

# -----------------------------------------------------------------
# Step 4: Send the request to the model server
# -----------------------------------------------------------------
print("\nSending request to deployed model at", URL)
response = requests.post(
    URL,
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload),
)

# -----------------------------------------------------------------
# Step 5: Parse and display the response
# -----------------------------------------------------------------
if response.status_code == 200:
    result = response.json()
    predictions = result["predictions"]

    print("\n" + "=" * 60)
    print("Predictions from deployed model:")
    print("=" * 60)
    for i, (pred, actual) in enumerate(zip(predictions, y_actual)):
        pred_label   = "Will rain"    if pred == 1   else "No rain"
        actual_label = "Will rain"    if actual == 1 else "No rain"
        match = "CORRECT" if pred == actual else "WRONG"
        print(f"  Day {i+1}: predicted={pred_label:<10} actual={actual_label:<10} [{match}]")
else:
    print(f"\nError {response.status_code}: {response.text}")