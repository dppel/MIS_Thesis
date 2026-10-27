"""
Phase 3: Publication-Grade Figures & Statistical Analysis for the Imbalance Trilemma Study.
Generates 300-DPI figures in research_paper/figures/:
  1. fig1_calibration_reliability_curves.png: Reliability diagrams across the 3 datasets (XGBoost & RF)
  2. fig2_ece_and_pd_distortion.png: Bar chart comparing Expected Calibration Error (ECE) Raw vs. Platt-Calibrated
  3. fig3_shap_stability_comparison.png: SHAP Rank Stability (Spearman rho & Cosine Sim) across SMOTE, ADASYN, Cost-Sensitive
  4. fig4_shap_feature_rank_shift_thesis.png: Top-10 Feature Importance comparison on the MSc Thesis dataset (XGBoost)
  5. fig5_trilemma_tradeoff_pareto.png: 3D/2D Trade-off scatter (Macro-F1 vs. Calibration ECE vs. SHAP Stability)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from scipy.stats import wilcoxon

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["figure.dpi"] = 300


def main():
    os.makedirs("research_paper/figures", exist_ok=True)
    df = pd.read_csv("research_paper/results/trilemma_benchmark_results.csv")

    with open("research_paper/results/calibration_predictions.json", "r", encoding="utf-8") as f:
        cal_data = json.load(f)

    with open("research_paper/results/shap_vectors.json", "r", encoding="utf-8") as f:
        shap_data = json.load(f)

    palette = {
        "Baseline": "#2b5c8f",
        "SMOTE": "#d95f02",
        "ADASYN": "#e7298a",
        "Cost-Sensitive": "#1b9e77",
    }
    markers = {
        "Baseline": "o",
        "SMOTE": "s",
        "ADASYN": "^",
        "Cost-Sensitive": "D",
    }

    # -------------------------------------------------------------------------
    # Figure 1: Reliability Diagrams (Calibration Curves) for XGBoost across 3 Datasets
    # Top row: Uncalibrated (Raw) | Bottom row: Post-Hoc Platt Calibrated
    # -------------------------------------------------------------------------
    datasets = list(cal_data.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)

    for col_idx, ds_name in enumerate(datasets):
        y_test = np.array(cal_data[ds_name]["y_test"])
        xgb_preds = cal_data[ds_name]["models"]["XGBoost"]

        for row_idx, (mode, mode_title) in enumerate([("raw", "Uncalibrated (Raw)"), ("platt", "Post-Hoc Platt Calibrated")]):
            ax = axes[row_idx, col_idx]
            ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="Perfect Calibration")

            for strat in ["Baseline", "SMOTE", "ADASYN", "Cost-Sensitive"]:
                probs = np.array(xgb_preds[strat][mode])
                prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=8, strategy="uniform")
                ax.plot(
                    prob_pred,
                    prob_true,
                    marker=markers[strat],
                    color=palette[strat],
                    lw=1.8,
                    markersize=5,
                    label=strat,
                )

            if row_idx == 0:
                ax.set_title(f"{ds_name}\n({mode_title})", fontweight="bold")
            else:
                ax.set_title(f"({mode_title})", fontweight="bold")

            if row_idx == 1:
                ax.set_xlabel("Mean Predicted Probability (PD)")
            if col_idx == 0:
                ax.set_ylabel("Observed Default Fraction")
            ax.grid(True, linestyle=":", alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("research_paper/figures/fig1_calibration_reliability_curves.png", bbox_inches="tight")
    plt.close()
    print("Saved fig1_calibration_reliability_curves.png")

    # -------------------------------------------------------------------------
    # Figure 2: Expected Calibration Error (ECE) Before vs. After Platt Scaling
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    sns.barplot(
        data=df,
        x="Model",
        y="ECE_Raw",
        hue="Strategy",
        palette=palette,
        errorbar=None,
        ax=axes[0],
    )
    axes[0].set_title("A. Uncalibrated Expected Calibration Error (ECE ↓)", fontweight="bold")
    axes[0].set_ylabel("Mean ECE across Datasets (Lower is Better)")
    axes[0].set_xlabel("")
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].get_legend().remove()

    sns.barplot(
        data=df,
        x="Model",
        y="ECE_Platt",
        hue="Strategy",
        palette=palette,
        errorbar=None,
        ax=axes[1],
    )
    axes[1].set_title("B. Post-Hoc Platt-Calibrated ECE (ECE ↓)", fontweight="bold")
    axes[1].set_ylabel("")
    axes[1].set_xlabel("")
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    axes[1].legend(title="Imbalance Strategy", frameon=True)

    plt.tight_layout()
    plt.savefig("research_paper/figures/fig2_ece_and_pd_distortion.png", bbox_inches="tight")
    plt.close()
    print("Saved fig2_ece_and_pd_distortion.png")

    # -------------------------------------------------------------------------
    # Figure 3: SHAP Explanation Stability (Spearman Rank Correlation & Cosine Sim)
    # -------------------------------------------------------------------------
    df_non_base = df[df["Strategy"] != "Baseline"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    sns.barplot(
        data=df_non_base,
        x="Model",
        y="SHAP_Spearman_Rho",
        hue="Strategy",
        palette=palette,
        errorbar=None,
        ax=axes[0],
    )
    axes[0].set_ylim(0.0, 1.05)
    axes[0].axhline(1.0, color="black", linestyle="--", lw=1, alpha=0.6)
    axes[0].set_title("A. SHAP Feature Rank Correlation vs. Baseline (Spearman ρ ↑)", fontweight="bold")
    axes[0].set_ylabel("Spearman Rank Correlation (ρ)")
    axes[0].set_xlabel("")
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].get_legend().remove()

    sns.barplot(
        data=df_non_base,
        x="Model",
        y="SHAP_Cosine_Sim",
        hue="Strategy",
        palette=palette,
        errorbar=None,
        ax=axes[1],
    )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].axhline(1.0, color="black", linestyle="--", lw=1, alpha=0.6)
    axes[1].set_title("B. SHAP Global Attribution Vector Cosine Similarity (↑)", fontweight="bold")
    axes[1].set_ylabel("Cosine Similarity vs. Baseline")
    axes[1].set_xlabel("")
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    axes[1].legend(title="Imbalance Strategy", loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig("research_paper/figures/fig3_shap_stability_comparison.png", bbox_inches="tight")
    plt.close()
    print("Saved fig3_shap_stability_comparison.png")

    # -------------------------------------------------------------------------
    # Figure 4: SHAP Feature Importance Shift on MSc Thesis Dataset (XGBoost)
    # -------------------------------------------------------------------------
    thesis_key = "Credit Card Approval (Thesis)"
    feat_names = [f.replace("num__", "").replace("cat__", "") for f in shap_data[thesis_key]["feature_names"]]
    xgb_shap = shap_data[thesis_key]["models"]["XGBoost"]

    shap_df = pd.DataFrame({"Feature": feat_names})
    for strat in ["Baseline", "SMOTE", "ADASYN", "Cost-Sensitive"]:
        vec = np.array(xgb_shap[strat])
        # Normalize to relative percentage of total |SHAP| for clean comparison
        shap_df[strat] = (vec / np.sum(vec)) * 100.0

    top10_features = shap_df.sort_values("Baseline", ascending=False).head(10)["Feature"].tolist()
    shap_top10 = shap_df[shap_df["Feature"].isin(top10_features)].copy()
    shap_top10["Feature"] = pd.Categorical(shap_top10["Feature"], categories=top10_features[::-1], ordered=True)
    shap_top10 = shap_top10.sort_values("Feature")

    shap_melted = shap_top10.melt(id_vars="Feature", var_name="Strategy", value_name="Relative_SHAP_Pct")

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=shap_melted,
        y="Feature",
        x="Relative_SHAP_Pct",
        hue="Strategy",
        palette=palette,
        orient="h",
    )
    plt.title(
        "Top-10 Global SHAP Attributions on Credit Card Approval Dataset (XGBoost)\n"
        "Notice how SMOTE/ADASYN inflate One-Hot Categorical Features vs. Continuous Features",
        fontweight="bold",
    )
    plt.xlabel("Relative Global Feature Importance (% of Total Mean |SHAP|)")
    plt.ylabel("Applicant Feature")
    plt.grid(axis="x", linestyle=":", alpha=0.5)
    plt.legend(title="Strategy", frameon=True)
    plt.tight_layout()
    plt.savefig("research_paper/figures/fig4_shap_feature_rank_shift_thesis.png", bbox_inches="tight")
    plt.close()
    print("Saved fig4_shap_feature_rank_shift_thesis.png")

    # -------------------------------------------------------------------------
    # Figure 5: The Imbalance Trilemma Trade-off Scatter
    # X = Macro-F1 (Discrimination), Y = 1 - ECE_Raw (Calibration Fidelity), Size/Color = Strategy
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8.5, 5.5))
    for strat in ["Baseline", "SMOTE", "ADASYN", "Cost-Sensitive"]:
        sub = df[df["Strategy"] == strat]
        plt.scatter(
            sub["Macro_F1"],
            sub["ECE_Raw"],
            c=palette[strat],
            marker=markers[strat],
            s=110,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.7,
            label=f"{strat} (Mean SHAP ρ = {sub['SHAP_Spearman_Rho'].mean():.2f})",
        )

    plt.title("The Credit Scoring Imbalance Trilemma: Macro-F1 vs. Calibration Error vs. SHAP Stability", fontweight="bold")
    plt.xlabel("Test Macro-F1 Score (Higher = Better Balanced Classification ↑)")
    plt.ylabel("Uncalibrated Expected Calibration Error — ECE (Lower = Better PD Calibration ↓)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(title="Strategy & Explanation Fidelity", frameon=True)
    plt.tight_layout()
    plt.savefig("research_paper/figures/fig5_trilemma_tradeoff_pareto.png", bbox_inches="tight")
    plt.close()
    print("Saved fig5_trilemma_tradeoff_pareto.png")

    # Print Aggregated Summary Table for Manuscript
    summary = (
        df.groupby("Strategy")[
            [
                "ROC_AUC",
                "PR_AUC",
                "Macro_F1",
                "Minority_Recall",
                "ECE_Raw",
                "ECE_Platt",
                "Brier_Raw",
                "Brier_Platt",
                "SHAP_Spearman_Rho",
                "SHAP_Cosine_Sim",
                "SHAP_Top5_Overlap",
            ]
        ]
        .mean()
        .round(4)
    )
    print("\n=== AGGREGATED MEAN METRICS ACROSS ALL DATASETS & MODELS ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
