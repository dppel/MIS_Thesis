"""
Unified Dataset Loader for the Credit Scoring Imbalance Trilemma Study.
Loads three public credit risk datasets with varying sample sizes and class imbalance ratios:
1. Credit Card Approval (MSc Thesis dataset: application_record.csv + credit_record.csv)
2. Taiwan Credit Card Default (UCI / OpenML ID 42477, N=30,000)
3. German Credit Risk (Statlog UCI / OpenML ID 31, N=1,000)

Convention across all datasets:
    y = 1 -> Default / Severe Delinquency (Minority class — Positive class for PD modeling)
    y = 0 -> Non-Default / Good Standing (Majority class)
"""

import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml


def load_thesis_credit_card_dataset(data_dir: str = "./data"):
    """
    Loads and aggregates the MSc Thesis Credit Card Approval dataset.
    Customer-level aggregation prevents data leakage across monthly records.
    Here, y = 1 denotes severe delinquency observed (STATUS in 1..5),
    and y = 0 denotes good standing (STATUS in C, X, 0).
    """
    app_path = os.path.join(data_dir, "application_record.csv")
    cred_path = os.path.join(data_dir, "credit_record.csv")

    application = pd.read_csv(app_path).drop_duplicates(subset="ID", keep="first").copy()
    credit = pd.read_csv(cred_path)

    credit["STATUS"] = credit["STATUS"].astype(str).str.strip()
    valid_statuses = {"0", "1", "2", "3", "4", "5", "C", "X"}
    credit = credit[credit["STATUS"].isin(valid_statuses)].copy()

    # 1 = Severe delinquency (30+ days past due: statuses 1-5), 0 = Good standing (C, X, 0)
    credit["severe_delinquency"] = credit["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)

    customer_target = (
        credit.groupby("ID", as_index=False)["severe_delinquency"]
        .max()
        .rename(columns={"severe_delinquency": "TARGET_DEFAULT"})
    )

    data = application.merge(customer_target, on="ID", how="inner")

    # Feature engineering from thesis
    data["AGE_YEARS"] = -data["DAYS_BIRTH"] / 365.25
    data.loc[data["DAYS_EMPLOYED"] > 0, "DAYS_EMPLOYED"] = np.nan
    data["YEARS_EMPLOYED"] = -data["DAYS_EMPLOYED"] / 365.25

    drop_columns = ["ID", "DAYS_BIRTH", "DAYS_EMPLOYED", "TARGET_DEFAULT"]
    X = data.drop(columns=drop_columns, errors="ignore")
    y = data["TARGET_DEFAULT"].astype(int)

    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    if constant_cols:
        X = X.drop(columns=constant_cols)

    return X, y, "Credit Card Approval (Thesis)"


def load_taiwan_credit_default():
    """
    Loads the UCI Taiwan Default of Credit Card Clients dataset (N=30,000, 23 features).
    y = 1 -> Default next month (~22.12%)
    y = 0 -> Non-default (~77.88%)
    """
    ds = fetch_openml(data_id=42477, as_frame=True, parser="auto")
    X = ds.data.copy()
    y = ds.target.astype(int)

    # Ensure categorical/integer columns are properly typed as numeric
    for col in X.columns:
        try:
            X[col] = pd.to_numeric(X[col])
        except (ValueError, TypeError):
            pass

    return X, y, "Taiwan Credit Default (UCI)"


def load_german_credit():
    """
    Loads the Statlog German Credit dataset (N=1,000, 20 features).
    y = 1 -> Bad credit risk / Default (30.0%)
    y = 0 -> Good credit risk (70.0%)
    """
    ds = fetch_openml(data_id=31, as_frame=True, parser="auto")
    X = ds.data.copy()
    # Target in OpenML 31 is 'bad' vs 'good'
    y = (ds.target.astype(str).str.lower() == "bad").astype(int)

    # Convert categorical dtype columns to object/string for standard SimpleImputer + OneHotEncoder
    for col in X.select_dtypes(include=["category"]).columns:
        X[col] = X[col].astype(str)

    return X, y, "German Credit Risk (UCI)"


def get_all_datasets(data_dir: str = "./data"):
    datasets = [
        load_thesis_credit_card_dataset(data_dir),
        load_taiwan_credit_default(),
        load_german_credit(),
    ]
    return datasets


if __name__ == "__main__":
    for X, y, name in get_all_datasets():
        pos_rate = y.mean() * 100
        print(f"{name:32s} | Shape: {str(X.shape):14s} | Default Rate (y=1): {pos_rate:.2f}% ({y.sum()}/{len(y)})")
