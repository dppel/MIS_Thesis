"""
Independent Faithfulness / Perturbation Check (Ablation Test):
Tests whether the Top-3 features selected by each strategy's SHAP vector
are genuinely more informative on the real held-out test distribution.
For each strategy (SMOTE, ADASYN, Cost-Sensitive), we mask (mean-impute) its Top-3
SHAP features in X_test and measure the resulting drop in Baseline ROC-AUC (ΔAUC).
A larger ΔAUC drop proves higher objective faithfulness to the underlying model & data!
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN

from data_loader import get_all_datasets
from run_experiments import build_preprocessor, get_model, extract_global_shap_vector, RANDOM_STATE


def main():
    datasets = get_all_datasets("./data")
    models = ["Random Forest", "XGBoost", "LightGBM"]
    strategies = ["Baseline", "SMOTE", "ADASYN", "Cost-Sensitive"]

    rows = []
    for X, y, ds_name in datasets:
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
        )
        X_train, _, y_train, _ = train_test_split(
            X_train_full, y_train_full, test_size=0.20, stratify=y_train_full, random_state=RANDOM_STATE
        )

        prep = build_preprocessor(X_train)
        X_tr = prep.fit_transform(X_train)
        X_te = prep.transform(X_test)
        neg_pos = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

        rng = np.random.RandomState(RANDOM_STATE)
        shap_idx = rng.choice(len(X_te), size=min(300, len(X_te)), replace=False)
        X_shap = X_te[shap_idx]
        col_means = np.mean(X_tr, axis=0)

        for m_name in models:
            # Fit reference model on unmodified training data
            ref_clf = get_model(m_name, "Baseline", neg_pos)
            ref_clf.fit(X_tr, y_train.values)
            base_auc = roc_auc_score(y_test, ref_clf.predict_proba(X_te)[:, 1])

            for strat in strategies:
                if strat == "SMOTE":
                    X_f, y_f = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr, y_train.values)
                elif strat == "ADASYN":
                    try:
                        X_f, y_f = ADASYN(random_state=RANDOM_STATE).fit_resample(X_tr, y_train.values)
                    except Exception:
                        X_f, y_f = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr, y_train.values)
                else:
                    X_f, y_f = X_tr, y_train.values

                clf = get_model(m_name, strat, neg_pos)
                clf.fit(X_f, y_f)
                s_vec = extract_global_shap_vector(clf, X_shap)
                top3_idx = np.argsort(s_vec)[::-1][:3]

                # Mask Top-3 features identified by `strat` in the held-out test set
                X_te_masked = X_te.copy()
                for c_i in top3_idx:
                    X_te_masked[:, c_i] = col_means[c_i]

                masked_auc = roc_auc_score(y_test, ref_clf.predict_proba(X_te_masked)[:, 1])
                auc_drop = base_auc - masked_auc
                rows.append({
                    "Dataset": ds_name,
                    "Model": m_name,
                    "Strategy": strat,
                    "Base_AUC": round(base_auc, 4),
                    "Masked_Top3_AUC": round(masked_auc, 4),
                    "AUC_Drop_Top3_Masking": round(auc_drop, 4),
                })

    df_ab = pd.DataFrame(rows)
    df_ab.to_csv("research_paper/results/shap_ablation_faithfulness.csv", index=False)
    print("=== INDEPENDENT TOP-3 FEATURE ABLATION FAITHFULNESS (MEAN AUC DROP WHEN MASKED) ===")
    print(df_ab.groupby("Strategy")[["Base_AUC", "Masked_Top3_AUC", "AUC_Drop_Top3_Masking"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
