#!/usr/bin/env python3
"""
Usage:
  - Produces model_1.pickle (LinearRegression on 100g_USD)
  - Produces model_2.pickle (DecisionTreeRegressor on 100g_USD + roast_cat)
"""

import sys
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Default CSV URL.
COFFEE_CSV = os.environ.get("COFFEE_CSV") or (
    "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
)

def roast_category(val):
    """
    Map roast values to numeric labels.

    Rules:
    - If val is missing (NaN or None) -> return np.nan
    - If val is a known string -> return an integer label
    - If val is something else, attempt to convert to string and map

    This function intentionally returns simple integer labels; the specific
    mapping is created from the unique roast values found in the dataset.
    """
    if val is None:
        return np.nan
    if (isinstance(val, float) and np.isnan(val)):
        return np.nan
    # Keep string conversion defensive
    return str(val)

def build_roast_mapping(series):
    """
    Given a pandas Series of roast values, build a mapping from roast string -> int.
    Missing values remain np.nan.
    """
    unique_vals = sorted({str(x) for x in series.dropna().unique()})
    mapping = {v: i + 1 for i, v in enumerate(unique_vals)}  # start labels at 1
    return mapping

def main(csv_url):
    # Load data
    df = pd.read_csv(csv_url)

    # Ensure required columns exist
    required = {"rating", "100g_USD", "roast"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # ------ Exercise 1: Linear Regression on 100g_USD ------
    # Prepare X,y for linear regression
    # Drop rows where 100g_USD or rating are missing
    lr_df = df[["100g_USD", "rating"]].copy()
    lr_df = lr_df.dropna(subset=["100g_USD", "rating"])
    X_lr = lr_df[["100g_USD"]].values.reshape(-1, 1)
    y_lr = lr_df["rating"].values

    lr = LinearRegression()
    lr.fit(X_lr, y_lr)

    # Save model_1.pickle
    with open("model_1.pickle", "wb") as f:
        pickle.dump(lr, f)

    # ------ Exercise 2: Decision Tree Regressor on 100g_USD + roast_cat ------
    # Create roast_cat numeric mapping
    # Apply roast_category to normalize values, then map to ints
    roast_raw = df["roast"].apply(roast_category)
    roast_map = build_roast_mapping(roast_raw)

    # Create roast_cat where known values map to integers, missing -> np.nan
    def _map_roast_to_int(x):
        if x is None:
            return np.nan
        if isinstance(x, float) and np.isnan(x):
            return np.nan
        sx = str(x)
        return roast_map.get(sx, np.nan)

    df["roast_cat"] = roast_raw.apply(_map_roast_to_int)

    # Prepare training data: use rows where 100g_USD and rating are present
    # roast_cat can be NaN; DecisionTreeRegressor in scikit-learn doesn't accept NaN,
    # so we'll drop rows where roast_cat is NaN for training simplicity.
    dtr_df = df[["100g_USD", "roast_cat", "rating"]].copy()
    dtr_df = dtr_df.dropna(subset=["100g_USD", "roast_cat", "rating"])
    X_dtr = dtr_df[["100g_USD", "roast_cat"]].values
    y_dtr = dtr_df["rating"].values

    dtr = DecisionTreeRegressor(random_state=42)
    dtr.fit(X_dtr, y_dtr)

    # Save model_2.pickle
    with open("model_2.pickle", "wb") as f:
        pickle.dump(dtr, f)

    # Print summary
    print("Saved model_1.pickle (LinearRegression on 100g_USD)")
    print("Saved model_2.pickle (DecisionTreeRegressor on 100g_USD and roast_cat)")
    print(f"roast mapping example (first 10): {dict(list(roast_map.items())[:10])}")

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else COFFEE_CSV
    main(url)
