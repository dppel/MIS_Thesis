"""
Independent Top-3 Feature-Masking Protocol (Dual Evaluation):
Evaluates the predictive signal of the Top-3 features selected by each strategy's SHAP vector under:
1. Self-Model Masking (ΔAUC_self): Measures the ROC-AUC drop on the strategy's OWN trained classifier
   when its Top-3 SHAP features are masked (replaced by training mean) in X_test.
2. Common Reference Masking (ΔAUC_ref): Measures the ROC-AUC drop on the fixed unweighted Baseline
   evaluator when those same Top-3 features are masked in X_test.
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
            # Fit common reference model on unmodified training data
            ref_clf = get_model(m_name, "Baseline", neg_pos)
            ref_clf.fit(X_tr, y_train.values)
            ref_full_auc = roc_auc_score(y_test, ref_clf.predict_proba(X_te)[:, 1])

            for strat in strategies:
                if strat == "SMOTE":
                    X_f, y_f = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr, y_train.values)
                elif strat == "ADASYN":
                    X_f, y_f = ADASYN(random_state=RANDOM_STATE).fit_resample(X_tr, y_train.values)
                else:
                    X_f, y_f = X_tr, y_train.values

                clf = get_model(m_name, strat, neg_pos)
                clf.fit(X_f, y_f)
                self_full_auc = roc_auc_score(y_test, clf.predict_proba(X_te)[:, 1])

                s_vec = extract_global_shap_vector(clf, X_shap)
                top3_idx = np.argsort(s_vec)[::-1][:3]

                # Mask Top-3 features identified by `strat` in the held-out test set
                X_te_masked = X_te.copy()
                for c_i in top3_idx:
                    X_te_masked[:, c_i] = col_means[c_i]

                # 1. Self-Model evaluation (drop in the strategy's own model AUC)
                self_masked_auc = roc_auc_score(y_test, clf.predict_proba(X_te_masked)[:, 1])
                self_auc_drop = self_full_auc - self_masked_auc

                # 2. Common Reference evaluation (drop in the unweighted reference model AUC)
                ref_masked_auc = roc_auc_score(y_test, ref_clf.predict_proba(X_te_masked)[:, 1])
                ref_auc_drop = ref_full_auc - ref_masked_auc

                rows.append({
                    "Dataset": ds_name,
                    "Model": m_name,
                    "Strategy": strat,
                    "Self_Full_AUC": round(self_full_auc, 4),
                    "Self_Masked_Top3_AUC": round(self_masked_auc, 4),
                    "Self_AUC_Drop_Top3": round(self_auc_drop, 4),
                    "Ref_Full_AUC": round(ref_full_auc, 4),
                    "Ref_Masked_Top3_AUC": round(ref_masked_auc, 4),
                    "Ref_AUC_Drop_Top3": round(ref_auc_drop, 4),
                })

    df_ab = pd.DataFrame(rows)
    df_ab.to_csv("research_paper/results/shap_ablation_faithfulness.csv", index=False)
    print("=== DUAL TOP-3 FEATURE MASKING EVALUATION (SELF-MODEL vs. COMMON REFERENCE) ===")
    print(
        df_ab.groupby("Strategy")[
            ["Self_Full_AUC", "Self_Masked_Top3_AUC", "Self_AUC_Drop_Top3", "Ref_Full_AUC", "Ref_Masked_Top3_AUC", "Ref_AUC_Drop_Top3"]
        ]
        .mean()
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
