"""
Phase 2: Experimental Pipeline for the Imbalance Trilemma in Credit Scoring.
Evaluates:
  - 3 Datasets: Credit Card Approval (Thesis), Taiwan Credit Default (UCI), German Credit Risk (UCI)
  - 4 Models: Decision Tree, Random Forest, XGBoost, LightGBM
  - 4 Imbalance Strategies: Baseline (None), SMOTE, ADASYN, Cost-Sensitive Learning
Across 3 Pillars:
  1. Discrimination: ROC-AUC, PR-AUC (Average Precision), Macro-F1, Minority F1, Minority Recall
  2. Probability Calibration: Brier Score & Expected Calibration Error (ECE) — Uncalibrated vs. Platt Scaling vs. Prior-Shift Bayes Correction
  3. SHAP Explanation Stability: Spearman Rank Correlation (rho), Cosine Similarity, Top-5 Feature Overlap vs. Baseline
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.special import logit, expit

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    brier_score_loss,
)

from imblearn.over_sampling import SMOTE, ADASYN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

from data_loader import get_all_datasets

RANDOM_STATE = 1024


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) with uniform probability bins in [0, 1].
    ECE = sum_{b=1}^B (|I_b| / N) * |acc(I_b) - conf(I_b)|
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = binids == b
        if np.any(mask):
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def prior_shift_correction(y_prob_resampled: np.ndarray, pi_true: float, pi_train: float = 0.5) -> np.ndarray:
    """
    Analytic Elkan (2001) / Saerens et al. (2002) Bayes prior-shift probability correction
    for models trained on balanced (pi_train=0.5) oversampled data:
        logit(P_corrected) = logit(P_resampled) + log(pi_true / (1 - pi_true)) - log(pi_train / (1 - pi_train))
    """
    eps = 1e-6
    p_clip = np.clip(y_prob_resampled, eps, 1.0 - eps)
    log_odds = logit(p_clip) + np.log(pi_true / (1.0 - pi_true)) - np.log(pi_train / (1.0 - pi_train))
    return expit(log_odds)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    num_pipe = SklearnPipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = SklearnPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [("num", num_pipe, numeric_features)]
    if categorical_features:
        transformers.append(("cat", cat_pipe, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_model(model_name: str, strategy: str, neg_pos_ratio: float):
    """
    Returns tree-based classifier configured with or without Cost-Sensitive weights.
    Uses balanced regularized depth so probabilities and SHAP values are well-behaved across datasets.
    """
    is_cost_sensitive = (strategy == "Cost-Sensitive")
    class_weight = "balanced" if is_cost_sensitive else None
    scale_weight = neg_pos_ratio if is_cost_sensitive else 1.0

    if model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=5,
            class_weight=class_weight,
            random_state=RANDOM_STATE,
        )
    elif model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=3,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.05,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def extract_global_shap_vector(model, X_sample: np.ndarray) -> np.ndarray:
    """
    Computes mean(|SHAP|) vector of shape (n_features,) for positive class (y=1).
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample, check_additivity=False)

    if isinstance(sv, list):
        # List of arrays per class -> class 1
        sv_pos = sv[1]
    elif isinstance(sv, np.ndarray) and sv.ndim == 3:
        # Shape (n_samples, n_features, 2)
        sv_pos = sv[:, :, 1]
    else:
        sv_pos = sv

    return np.mean(np.abs(sv_pos), axis=0)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main():
    os.makedirs("research_paper/results", exist_ok=True)
    datasets = get_all_datasets("./data")

    models = ["Decision Tree", "Random Forest", "XGBoost", "LightGBM"]
    strategies = ["Baseline", "SMOTE", "ADASYN", "Cost-Sensitive"]

    all_rows = []
    calibration_curves_data = {}
    shap_vectors_data = {}

    for X, y, dataset_name in datasets:
        print(f"\n=======================================================")
        print(f"Dataset: {dataset_name} | N={len(y)}, Default Rate={y.mean()*100:.2f}%")
        print(f"=======================================================")

        # 70/30 Stratified Split (further split 70% train into 80% fit / 20% calibration validation)
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
        )
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train_full, y_train_full, test_size=0.20, stratify=y_train_full, random_state=RANDOM_STATE
        )

        preprocessor = build_preprocessor(X_train)
        X_train_trans = preprocessor.fit_transform(X_train)
        X_cal_trans = preprocessor.transform(X_cal)
        X_test_trans = preprocessor.transform(X_test)
        feature_names = preprocessor.get_feature_names_out().tolist()

        pi_true = float(y_train.mean())
        neg_pos_ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

        # Fixed test subsample for consistent SHAP comparison across strategies
        rng = np.random.RandomState(RANDOM_STATE)
        shap_idx = rng.choice(len(X_test_trans), size=min(400, len(X_test_trans)), replace=False)
        X_shap_sample = X_test_trans[shap_idx]

        calibration_curves_data[dataset_name] = {
            "y_test": y_test.values.tolist(),
            "models": {},
        }
        shap_vectors_data[dataset_name] = {
            "feature_names": feature_names,
            "models": {},
        }

        for model_name in models:
            calibration_curves_data[dataset_name]["models"][model_name] = {}
            shap_vectors_data[dataset_name]["models"][model_name] = {}
            baseline_shap_vec = None

            for strategy in strategies:
                print(f"  -> Running {model_name:15s} | Strategy: {strategy:15s} ...", end=" ", flush=True)

                if strategy == "SMOTE":
                    sampler = SMOTE(random_state=RANDOM_STATE)
                    X_tr_fit, y_tr_fit = sampler.fit_resample(X_train_trans, y_train.values)
                elif strategy == "ADASYN":
                    sampler = ADASYN(random_state=RANDOM_STATE)
                    X_tr_fit, y_tr_fit = sampler.fit_resample(X_train_trans, y_train.values)
                else:
                    X_tr_fit, y_tr_fit = X_train_trans, y_train.values

                pi_train_effective = float(np.mean(y_tr_fit)) if strategy != "Cost-Sensitive" else 0.5

                clf = get_model(model_name, strategy, neg_pos_ratio)
                clf.fit(X_tr_fit, y_tr_fit)

                # 1. Raw (uncalibrated) predictions on test set
                y_prob_raw = clf.predict_proba(X_test_trans)[:, 1]
                y_pred_raw = (y_prob_raw >= 0.5).astype(int)

                # 2. Post-hoc Platt Scaling (Sigmoid Calibration) on held-out calibration set (true distribution!)
                # In sklearn >=1.6, use FrozenEstimator or manual LogisticRegression on logit/probabilities
                from sklearn.linear_model import LogisticRegression
                cal_probs_val = np.clip(clf.predict_proba(X_cal_trans)[:, 1], 1e-6, 1 - 1e-6)
                platt_lr = LogisticRegression()
                platt_lr.fit(logit(cal_probs_val).reshape(-1, 1), y_cal.values)
                y_prob_platt = platt_lr.predict_proba(
                    logit(np.clip(y_prob_raw, 1e-6, 1 - 1e-6)).reshape(-1, 1)
                )[:, 1]

                # 3. Analytic Prior-Shift Bayes Correction
                if strategy in ("SMOTE", "ADASYN", "Cost-Sensitive"):
                    y_prob_bayes = prior_shift_correction(y_prob_raw, pi_true=pi_true, pi_train=pi_train_effective)
                else:
                    y_prob_bayes = y_prob_raw

                # Compute Discrimination Metrics
                roc_auc = roc_auc_score(y_test, y_prob_raw)
                pr_auc = average_precision_score(y_test, y_prob_raw)
                acc = accuracy_score(y_test, y_pred_raw)
                macro_f1 = f1_score(y_test, y_pred_raw, average="macro", zero_division=0)
                min_f1 = f1_score(y_test, y_pred_raw, pos_label=1, zero_division=0)
                min_rec = recall_score(y_test, y_pred_raw, pos_label=1, zero_division=0)
                min_prec = precision_score(y_test, y_pred_raw, pos_label=1, zero_division=0)

                # Compute Calibration Metrics
                brier_raw = brier_score_loss(y_test, y_prob_raw)
                ece_raw = compute_ece(y_test.values, y_prob_raw)
                mean_pred_pd_raw = float(np.mean(y_prob_raw))

                brier_platt = brier_score_loss(y_test, y_prob_platt)
                ece_platt = compute_ece(y_test.values, y_prob_platt)
                mean_pred_pd_platt = float(np.mean(y_prob_platt))

                brier_bayes = brier_score_loss(y_test, y_prob_bayes)
                ece_bayes = compute_ece(y_test.values, y_prob_bayes)

                # Compute SHAP Explanation Stability vs. Baseline
                shap_vec = extract_global_shap_vector(clf, X_shap_sample)
                if strategy == "Baseline":
                    baseline_shap_vec = shap_vec.copy()
                    shap_spearman = 1.0
                    shap_cosine = 1.0
                    top5_overlap = 1.0
                else:
                    rho, _ = spearmanr(baseline_shap_vec, shap_vec)
                    shap_spearman = float(rho) if not np.isnan(rho) else 0.0
                    shap_cosine = cosine_sim(baseline_shap_vec, shap_vec)
                    top5_base = set(np.argsort(baseline_shap_vec)[::-1][:5])
                    top5_curr = set(np.argsort(shap_vec)[::-1][:5])
                    top5_overlap = len(top5_base.intersection(top5_curr)) / 5.0

                shap_vectors_data[dataset_name]["models"][model_name][strategy] = shap_vec.tolist()
                calibration_curves_data[dataset_name]["models"][model_name][strategy] = {
                    "raw": y_prob_raw.tolist(),
                    "platt": y_prob_platt.tolist(),
                    "bayes": y_prob_bayes.tolist(),
                }

                row = {
                    "Dataset": dataset_name,
                    "True_Default_Rate": round(pi_true, 4),
                    "Model": model_name,
                    "Strategy": strategy,
                    "ROC_AUC": round(roc_auc, 4),
                    "PR_AUC": round(pr_auc, 4),
                    "Accuracy": round(acc, 4),
                    "Macro_F1": round(macro_f1, 4),
                    "Minority_F1": round(min_f1, 4),
                    "Minority_Recall": round(min_rec, 4),
                    "Minority_Precision": round(min_prec, 4),
                    "Mean_Pred_PD_Raw": round(mean_pred_pd_raw, 4),
                    "Brier_Raw": round(brier_raw, 4),
                    "ECE_Raw": round(ece_raw, 4),
                    "Mean_Pred_PD_Platt": round(mean_pred_pd_platt, 4),
                    "Brier_Platt": round(brier_platt, 4),
                    "ECE_Platt": round(ece_platt, 4),
                    "Brier_Bayes": round(brier_bayes, 4),
                    "ECE_Bayes": round(ece_bayes, 4),
                    "SHAP_Spearman_Rho": round(shap_spearman, 4),
                    "SHAP_Cosine_Sim": round(shap_cosine, 4),
                    "SHAP_Top5_Overlap": round(top5_overlap, 2),
                }
                all_rows.append(row)
                print(
                    f"AUC={roc_auc:.3f} | MacroF1={macro_f1:.3f} | "
                    f"ECE(Raw->Platt)={ece_raw:.3f}->{ece_platt:.3f} | "
                    f"SHAP_rho={shap_spearman:.3f}"
                )

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv("research_paper/results/trilemma_benchmark_results.csv", index=False)

    with open("research_paper/results/shap_vectors.json", "w", encoding="utf-8") as f:
        json.dump(shap_vectors_data, f)

    with open("research_paper/results/calibration_predictions.json", "w", encoding="utf-8") as f:
        json.dump(calibration_curves_data, f)

    print("\nSaved all 48 experimental configurations to research_paper/results/trilemma_benchmark_results.csv")


if __name__ == "__main__":
    main()
