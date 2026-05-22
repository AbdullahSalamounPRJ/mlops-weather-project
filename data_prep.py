"""
Step 1: Data Preparation
=========================
Loads weatherAUS.csv, cleans it, and splits it into train/test sets.
Run this ONCE. Then train.py uses the output files.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------
# Step 1: Load the raw CSV
# -----------------------------------------------------------------
print("Loading raw data...")
df = pd.read_csv("weatherAUS.csv")
print(f"   Rows: {len(df):,}")
print(f"   Columns: {len(df.columns)}")

# -----------------------------------------------------------------
# Step 2: Drop the target rows that are missing
# -----------------------------------------------------------------
# If RainTomorrow is missing, we can't use that row for training.
df = df.dropna(subset=["RainTomorrow"])
print(f"\nAfter dropping rows with missing target: {len(df):,}")

# -----------------------------------------------------------------
# Step 3: Drop columns we don't need
# -----------------------------------------------------------------
# Date isn't directly useful as a feature (we'd need feature engineering).
# Evaporation/Sunshine/Cloud columns have ~50% missing - drop them.
columns_to_drop = ["Date", "Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"]
df = df.drop(columns=columns_to_drop)
print(f"\nDropped columns: {columns_to_drop}")

# -----------------------------------------------------------------
# Step 4: Handle missing values in the remaining columns
# -----------------------------------------------------------------
# For numeric columns -> fill missing with the MEDIAN (typical value)
# For text columns    -> fill missing with the MODE (most common value)
numeric_cols = df.select_dtypes(include=[np.number]).columns
text_cols    = df.select_dtypes(include=["object"]).columns

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

for col in text_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print(f"\nFilled missing values:")
print(f"   - {len(numeric_cols)} numeric columns -> filled with median")
print(f"   - {len(text_cols)} text columns -> filled with mode")

# -----------------------------------------------------------------
# Step 5: Convert text columns to numbers (label encoding)
# -----------------------------------------------------------------
# ML models only understand numbers, not text like "Sydney" or "Yes".
# LabelEncoder maps each unique text value to a unique integer.
# Example: "Sydney"->0, "Melbourne"->1, "Brisbane"->2, etc.
print(f"\nEncoding text columns to numbers...")
for col in text_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    print(f"   - {col}: encoded")

# -----------------------------------------------------------------
# Step 6: Separate features (X) from target (y)
# -----------------------------------------------------------------
# X = all columns EXCEPT RainTomorrow (these are the "inputs")
# y = the RainTomorrow column (this is what we want to PREDICT)
X = df.drop(columns=["RainTomorrow"])
y = df["RainTomorrow"]

print(f"\nFeature columns ({len(X.columns)}):")
print(f"   {list(X.columns)}")
print(f"\nTarget: RainTomorrow")
print(f"   - No (0): {(y == 0).sum():,}")
print(f"   - Yes (1): {(y == 1).sum():,}")

# -----------------------------------------------------------------
# Step 7: Split into training and testing sets
# -----------------------------------------------------------------
# 80% train -> model learns from this
# 20% test  -> model is evaluated on this (never seen during training!)
# stratify=y -> ensures both sets have the same proportion of Yes/No
# random_state=42 -> guarantees the same split every time (reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------------------------------------------
# Step 8: Save the cleaned data to CSV files
# -----------------------------------------------------------------
# We save both X and y combined, so we have everything in one file.
train_df = X_train.copy()
train_df["RainTomorrow"] = y_train

test_df = X_test.copy()
test_df["RainTomorrow"] = y_test

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"\nSaved:")
print(f"   - train.csv ({len(train_df):,} rows)")
print(f"   - test.csv  ({len(test_df):,} rows)")
print(f"\nData prep complete. Now run: python train.py")