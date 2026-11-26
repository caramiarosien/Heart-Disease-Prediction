"""Step 3: External Validation, Shift Diagnostics & Publication-Quality Figures.

Implements the locked external validation and shift-diagnostic protocol from
MASTERPLAN.md:
- Loads the 4 frozen Cleveland-trained `.joblib` pipelines (no retraining, no
  threshold tuning; fixed classification threshold = 0.5).
- Evaluates each frozen pipeline on the 3 independent external hospitals
  (Hungary, Switzerland, Long Beach VA) plus a pooled external benchmark (n=617),
  computing point estimates and 1,000-sample stratified bootstrap 95% CIs.
- Merges Cleveland internal 5-fold CV OOF results into a single unified table
  (`external_validation_metrics.csv` and `.md`).
- Runs four pre-specified shift diagnostics:
    1. Calibration Gap (`mean_pred_risk - observed_prevalence`) per hospital/model
    2. Feature Reliance vs. External Missingness (`shift_diagnostics_feature_reliance.csv`)
    3. Covariate Shift via Standardized Mean Difference (`shift_diagnostics_covariate_smd.csv`)
    4. Data-Hygiene Sensitivity Check for uncleaned impossible zeros (`chol_zero_sensitivity.csv`)
- Generates 3 colorblind-safe (Okabe-Ito), 300 DPI publication-quality figures:
    - `results/figures/fig1_cohort_audit.png`
    - `results/figures/fig2_external_performance.png`
    - `results/figures/fig3_shift_diagnostics.png`
"""

from __future__ import annotations

import joblib
import matplotlib

# Use non-interactive backend for reproducible headless rendering on macOS/Linux
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.pipeline import Pipeline

from common import (
    BINARY_FEATURES,
    COHORT_FILES,
    CONTINUOUS_FEATURES,
    EXCLUDED_PORTABILITY_FEATURES,
    FIGURES_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    TABLES_DIR,
    get_feature_set,
    load_cohort,
)
from metrics import METRIC_NAMES, bootstrap_metrics, compute_metrics, df_to_markdown

# Fixed evaluation hyperparameters (MASTERPLAN.md Sections 2.D & 2.E)
THRESHOLD = 0.5
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95

# Pre-specified 2 x 2 model grid in canonical display order
MODEL_SPECS: list[tuple[str, str, str]] = [
    ("full_13", "logreg", "Full-13 + LogReg"),
    ("full_13", "rf", "Full-13 + RF"),
    ("core_10", "logreg", "Core-10 + LogReg"),
    ("core_10", "rf", "Core-10 + RF"),
]

# External hospital ordering
EXTERNAL_HOSPITAL_KEYS = ["hungarian", "switzerland", "va"]

# Features evaluated for Standardized Mean Difference (continuous + binary)
SMD_FEATURES = [
    "age",
    "sex",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "exang",
    "fbs",
]

# Colorblind-safe Okabe-Ito palette
OKABE_ITO = {
    "blue": "#0072B2",
    "teal": "#009E73",
    "vermilion": "#D55E00",
    "amber": "#E69F00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "charcoal": "#222222",
    "gray": "#666666",
    "light_gray": "#E6E6E6",
}

MODEL_STYLES: dict[str, dict[str, str | float]] = {
    "Full-13 + LogReg": {
        "color": OKABE_ITO["blue"],
        "marker": "o",
        "short": "Full-13 LogReg",
    },
    "Core-10 + LogReg": {
        "color": OKABE_ITO["teal"],
        "marker": "s",
        "short": "Core-10 LogReg",
    },
    "Full-13 + RF": {
        "color": OKABE_ITO["vermilion"],
        "marker": "^",
        "short": "Full-13 RF",
    },
    "Core-10 + RF": {
        "color": OKABE_ITO["amber"],
        "marker": "D",
        "short": "Core-10 RF",
    },
}


# ---------------------------------------------------------------------------
# 1. Model Loading & External Evaluation
# ---------------------------------------------------------------------------


def load_frozen_models() -> dict[tuple[str, str], Pipeline]:
    """Load the 4 frozen Cleveland-trained scikit-learn pipelines from disk."""
    models: dict[tuple[str, str], Pipeline] = {}
    for feature_set, model_name, _ in MODEL_SPECS:
        path = MODELS_DIR / f"{feature_set}_{model_name}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"Expected frozen model at {path}. Run src/02_train_internal.py first."
            )
        models[(feature_set, model_name)] = joblib.load(path)
    return models


def load_internal_cleveland_rows(cleveland_df: pd.DataFrame) -> list[dict]:
    """Read Cleveland 5-fold CV OOF metrics into the unified schema."""
    internal_path = TABLES_DIR / "internal_cv_metrics.csv"
    internal_df = pd.read_csv(internal_path)

    n_patients = int(len(cleveland_df))
    n_disease = int(cleveland_df["target"].sum())
    n_healthy = n_patients - n_disease

    rows: list[dict] = []
    for feature_set, model_name, model_label in MODEL_SPECS:
        match = internal_df[
            (internal_df["feature_set"] == feature_set)
            & (internal_df["model"] == model_name)
        ]
        if len(match) != 1:
            raise RuntimeError(
                f"Expected 1 Cleveland CV row for {feature_set}/{model_name}, "
                f"found {len(match)}."
            )
        rec = match.iloc[0]

        row: dict[str, object] = {
            "cohort_key": "cleveland_cv",
            "hospital": "Cleveland (5-fold CV)",
            "evaluation_type": "internal_cv",
            "feature_set": feature_set,
            "model": model_name,
            "model_label": model_label,
            "n_patients": n_patients,
            "n_disease": n_disease,
            "n_healthy": n_healthy,
            "threshold": THRESHOLD,
        }
        for metric in METRIC_NAMES:
            row[metric] = float(rec[f"{metric}_oof"])
            row[f"{metric}_ci_low"] = float(rec[f"{metric}_oof_ci_low"])
            row[f"{metric}_ci_high"] = float(rec[f"{metric}_oof_ci_high"])

        obs_prev = float(rec["observed_prevalence_oof"])
        row["calibration_gap"] = float(rec["mean_pred_risk_oof"]) - obs_prev
        row["calibration_gap_ci_low"] = (
            float(rec["mean_pred_risk_oof_ci_low"]) - obs_prev
        )
        row["calibration_gap_ci_high"] = (
            float(rec["mean_pred_risk_oof_ci_high"]) - obs_prev
        )
        rows.append(row)

    return rows


def evaluate_cohort(
    cohort_key: str,
    hospital_label: str,
    evaluation_type: str,
    df: pd.DataFrame,
    models: dict[tuple[str, str], Pipeline],
) -> list[dict]:
    """Evaluate all 4 frozen pipelines on one cohort with stratified bootstrap CIs."""
    y_true = df["target"].to_numpy(dtype=int)
    n_patients = int(len(df))
    n_disease = int(y_true.sum())
    n_healthy = n_patients - n_disease

    rows: list[dict] = []
    for feature_set, model_name, model_label in MODEL_SPECS:
        features, _, _ = get_feature_set(feature_set)
        X = df[features].copy()

        pipe = models[(feature_set, model_name)]
        y_prob = pipe.predict_proba(X)[:, 1]

        boot = bootstrap_metrics(
            y_true=y_true,
            y_prob=y_prob,
            n_bootstrap=N_BOOTSTRAP,
            threshold=THRESHOLD,
            ci_level=CI_LEVEL,
            random_state=RANDOM_STATE,
        )

        row: dict[str, object] = {
            "cohort_key": cohort_key,
            "hospital": hospital_label,
            "evaluation_type": evaluation_type,
            "feature_set": feature_set,
            "model": model_name,
            "model_label": model_label,
            "n_patients": n_patients,
            "n_disease": n_disease,
            "n_healthy": n_healthy,
            "threshold": THRESHOLD,
        }
        for metric in METRIC_NAMES:
            row[metric] = boot[metric]["estimate"]
            row[f"{metric}_ci_low"] = boot[metric]["ci_low"]
            row[f"{metric}_ci_high"] = boot[metric]["ci_high"]

        # Stratified bootstrap preserves observed_prevalence exactly across replicates,
        # so the 95% CI for (mean_pred_risk - observed_prevalence) shifts identically.
        obs_prev = boot["observed_prevalence"]["estimate"]
        row["calibration_gap"] = boot["mean_pred_risk"]["estimate"] - obs_prev
        row["calibration_gap_ci_low"] = boot["mean_pred_risk"]["ci_low"] - obs_prev
        row["calibration_gap_ci_high"] = boot["mean_pred_risk"]["ci_high"] - obs_prev
        rows.append(row)

    return rows


def format_external_metrics_markdown(metrics_df: pd.DataFrame) -> str:
    """Create a readable Markdown table with point estimates and 95% bootstrap CIs."""

    def fmt_ci(row: pd.Series, metric: str) -> str:
        return (
            f"{row[metric]:.3f} "
            f"({row[f'{metric}_ci_low']:.3f}\u2013{row[f'{metric}_ci_high']:.3f})"
        )

    display_rows = []
    for _, row in metrics_df.iterrows():
        cal_gap = float(row["calibration_gap"])
        display_rows.append(
            {
                "Cohort": row["hospital"],
                "N": int(row["n_patients"]),
                "Model": row["model_label"],
                "ROC-AUC (95% CI)": fmt_ci(row, "roc_auc"),
                "PR-AUC (95% CI)": fmt_ci(row, "pr_auc"),
                "Balanced Acc (95% CI)": fmt_ci(row, "balanced_accuracy"),
                "Sensitivity (95% CI)": fmt_ci(row, "sensitivity"),
                "Specificity (95% CI)": fmt_ci(row, "specificity"),
                "Brier Score (95% CI)": fmt_ci(row, "brier_score"),
                "Mean Pred Risk": f"{row['mean_pred_risk']:.3f}",
                "Obs. Prev": f"{row['observed_prevalence']:.3f}",
                "Calib. Gap": f"{cal_gap:+.3f}",
                "Accuracy (Maj. Base)": (
                    f"{row['accuracy']:.3f} ({row['majority_baseline_acc']:.3f})"
                ),
            }
        )

    header = (
        "### Internal (Cleveland 5-Fold CV) & External Hospital Validation Metrics\n\n"
        "- **Protocol:** Frozen Cleveland-trained pipelines (`threshold = 0.50`, no "
        "retraining, no recalibration, impossible zeros `chol == 0` / `trestbps == 0` "
        "recoded to `NaN`).\n"
        "- **Uncertainty:** 1,000-sample stratified bootstrap 95% percentile CIs "
        "(`random_state = 42`).\n"
        "- **Calibration Gap:** `Mean Pred Risk - Observed Prevalence` (negative "
        "values indicate systematic under-prediction of disease risk).\n\n"
    )
    return header + df_to_markdown(pd.DataFrame(display_rows))


# ---------------------------------------------------------------------------
# 2. Shift Diagnostics
# ---------------------------------------------------------------------------


def compute_feature_reliance_table() -> pd.DataFrame:
    """Merge Cleveland Full-13 OOF permutation importance with external missingness."""
    imp_df = pd.read_csv(TABLES_DIR / "feature_importance_cleveland.csv")
    miss_df = pd.read_csv(TABLES_DIR / "missingness_by_hospital.csv")

    full_imp = imp_df[imp_df["feature_set"] == "full_13"].copy()
    logreg_imp = (
        full_imp[full_imp["model"] == "logreg"][
            ["feature", "perm_roc_auc_drop_mean", "perm_roc_auc_drop_sd"]
        ]
        .rename(
            columns={
                "perm_roc_auc_drop_mean": "logreg_perm_roc_auc_drop_mean",
                "perm_roc_auc_drop_sd": "logreg_perm_roc_auc_drop_sd",
            }
        )
        .reset_index(drop=True)
    )
    rf_imp = (
        full_imp[full_imp["model"] == "rf"][
            ["feature", "perm_roc_auc_drop_mean", "perm_roc_auc_drop_sd"]
        ]
        .rename(
            columns={
                "perm_roc_auc_drop_mean": "rf_perm_roc_auc_drop_mean",
                "perm_roc_auc_drop_sd": "rf_perm_roc_auc_drop_sd",
            }
        )
        .reset_index(drop=True)
    )

    miss_clean = miss_df.rename(
        columns={
            "Feature": "feature",
            "Feature Set": "feature_set_group",
            "Cleveland (%)": "cleveland_missing_pct",
            "Hungary (%)": "hungary_missing_pct",
            "Switzerland (%)": "switzerland_missing_pct",
            "Long Beach VA (%)": "va_missing_pct",
            "External Pooled (%)": "external_pooled_missing_pct",
        }
    )

    merged = (
        miss_clean.merge(logreg_imp, on="feature", how="inner")
        .merge(rf_imp, on="feature", how="inner")
        .sort_values("logreg_perm_roc_auc_drop_mean", ascending=False, ignore_index=True)
    )

    ordered_cols = [
        "feature",
        "feature_set_group",
        "logreg_perm_roc_auc_drop_mean",
        "logreg_perm_roc_auc_drop_sd",
        "rf_perm_roc_auc_drop_mean",
        "rf_perm_roc_auc_drop_sd",
        "cleveland_missing_pct",
        "hungary_missing_pct",
        "switzerland_missing_pct",
        "va_missing_pct",
        "external_pooled_missing_pct",
    ]
    return merged[ordered_cols]


def compute_smd(
    ref_series: pd.Series,
    ext_series: pd.Series,
    is_binary: bool,
) -> tuple[int, float, float, int, float, float, float]:
    """Compute Standardized Mean Difference (External minus Cleveland).

    Uses pooled sample SD (`ddof=1`) for continuous covariates and Austin's (2009)
    pooled Bernoulli variance `sqrt((p1*(1-p1) + p0*(1-p0)) / 2)` for binary
    covariates. Returns NaN cleanly if an external feature has <2 observed values
    (e.g., Switzerland serum cholesterol, which is 100% missing after cleaning).
    """
    ref_vals = ref_series.dropna().to_numpy(dtype=float)
    ext_vals = ext_series.dropna().to_numpy(dtype=float)

    n_ref = int(len(ref_vals))
    n_ext = int(len(ext_vals))

    ref_mean = float(np.mean(ref_vals)) if n_ref > 0 else np.nan
    ref_sd = float(np.std(ref_vals, ddof=1)) if n_ref > 1 else np.nan

    if n_ext < 2:
        ext_mean = float(np.mean(ext_vals)) if n_ext == 1 else np.nan
        return n_ref, ref_mean, ref_sd, n_ext, ext_mean, np.nan, np.nan

    ext_mean = float(np.mean(ext_vals))
    ext_sd = float(np.std(ext_vals, ddof=1))

    if is_binary:
        pooled_var = 0.5 * (
            ext_mean * (1.0 - ext_mean) + ref_mean * (1.0 - ref_mean)
        )
    else:
        pooled_var = 0.5 * (ext_sd**2 + ref_sd**2)

    if pooled_var <= 0.0:
        smd = np.nan
    else:
        smd = float((ext_mean - ref_mean) / np.sqrt(pooled_var))

    return n_ref, ref_mean, ref_sd, n_ext, ext_mean, ext_sd, smd


def compute_covariate_smd_table(
    cleveland_df: pd.DataFrame,
    external_cohorts: dict[str, tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Compute Standardized Mean Differences relative to Cleveland for each cohort."""
    rows = []
    for feature in SMD_FEATURES:
        is_binary = feature in BINARY_FEATURES
        feature_type = "binary" if is_binary else "continuous"
        assert is_binary or feature in CONTINUOUS_FEATURES

        for cohort_key, (hospital_label, ext_df) in external_cohorts.items():
            n_ref, ref_mean, ref_sd, n_ext, ext_mean, ext_sd, smd = compute_smd(
                cleveland_df[feature], ext_df[feature], is_binary=is_binary
            )
            rows.append(
                {
                    "feature": feature,
                    "feature_type": feature_type,
                    "cohort_key": cohort_key,
                    "hospital": hospital_label,
                    "cleveland_n": n_ref,
                    "cleveland_mean": ref_mean,
                    "cleveland_sd": ref_sd,
                    "external_n": n_ext,
                    "external_mean": ext_mean,
                    "external_sd": ext_sd,
                    "smd": smd,
                    "abs_smd": abs(smd) if pd.notna(smd) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def compute_chol_zero_sensitivity(
    models: dict[tuple[str, str], Pipeline],
) -> pd.DataFrame:
    """Diagnostic check: impact of leaving impossible zeros uncleaned in Switzerland & VA.

    Strictly a diagnostic sensitivity analysis; the primary protocol ALWAYS recodes
    `chol == 0` and `trestbps == 0` to `NaN` prior to Cleveland-fitted imputation.
    """
    rows = []
    diagnostic_cohorts = [
        ("switzerland", COHORT_FILES["switzerland"]["name"]),
        ("va", COHORT_FILES["va"]["name"]),
    ]

    for cohort_key, hospital_label in diagnostic_cohorts:
        df_cleaned = load_cohort(cohort_key, clean_impossible_zeros=True)
        df_uncleaned = load_cohort(cohort_key, clean_impossible_zeros=False)

        chol_zero_n = int((df_uncleaned["chol"] == 0).sum())
        trestbps_zero_n = int((df_uncleaned["trestbps"] == 0).sum())
        y_true = df_cleaned["target"].to_numpy(dtype=int)

        for feature_set, model_name, model_label in MODEL_SPECS:
            features, _, _ = get_feature_set(feature_set)
            pipe = models[(feature_set, model_name)]

            prob_cleaned = pipe.predict_proba(df_cleaned[features])[:, 1]
            prob_uncleaned = pipe.predict_proba(df_uncleaned[features])[:, 1]

            m_clean = compute_metrics(y_true, prob_cleaned, threshold=THRESHOLD)
            m_raw = compute_metrics(y_true, prob_uncleaned, threshold=THRESHOLD)

            max_abs_prob_diff = float(np.max(np.abs(prob_uncleaned - prob_cleaned)))
            mean_abs_prob_diff = float(np.mean(np.abs(prob_uncleaned - prob_cleaned)))

            rows.append(
                {
                    "hospital": hospital_label,
                    "cohort_key": cohort_key,
                    "n_patients": int(len(df_cleaned)),
                    "chol_zero_n": chol_zero_n,
                    "trestbps_zero_n": trestbps_zero_n,
                    "feature_set": feature_set,
                    "model": model_name,
                    "model_label": model_label,
                    "roc_auc_cleaned": m_clean["roc_auc"],
                    "roc_auc_uncleaned": m_raw["roc_auc"],
                    "delta_roc_auc": m_raw["roc_auc"] - m_clean["roc_auc"],
                    "balanced_acc_cleaned": m_clean["balanced_accuracy"],
                    "balanced_acc_uncleaned": m_raw["balanced_accuracy"],
                    "delta_balanced_acc": (
                        m_raw["balanced_accuracy"] - m_clean["balanced_accuracy"]
                    ),
                    "sensitivity_cleaned": m_clean["sensitivity"],
                    "sensitivity_uncleaned": m_raw["sensitivity"],
                    "specificity_cleaned": m_clean["specificity"],
                    "specificity_uncleaned": m_raw["specificity"],
                    "brier_cleaned": m_clean["brier_score"],
                    "brier_uncleaned": m_raw["brier_score"],
                    "delta_brier": m_raw["brier_score"] - m_clean["brier_score"],
                    "mean_pred_risk_cleaned": m_clean["mean_pred_risk"],
                    "mean_pred_risk_uncleaned": m_raw["mean_pred_risk"],
                    "delta_mean_pred_risk": (
                        m_raw["mean_pred_risk"] - m_clean["mean_pred_risk"]
                    ),
                    "mean_abs_prob_shift": mean_abs_prob_diff,
                    "max_abs_prob_shift": max_abs_prob_diff,
                    "diagnostic_note": (
                        "Diagnostic only (uncleaned zeros vs primary cleaned protocol)"
                    ),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Publication-Quality Figures (300 DPI, Okabe-Ito Palette)
# ---------------------------------------------------------------------------


def configure_matplotlib_style() -> None:
    """Apply clean, publication-grade typography and axis defaults."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10.5,
            "axes.titlesize": 11.8,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.0,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.9,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#222222",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
        }
    )


def make_fig1_cohort_audit() -> None:
    """Figure 1: Missingness heatmap across hospitals + disease prevalence bar chart."""
    miss_df = pd.read_csv(TABLES_DIR / "missingness_by_hospital.csv")
    cohort_df = pd.read_csv(TABLES_DIR / "cohort_summary.csv")

    # Order rows: Core-10 features on top, Full-13 only (`ca`, `thal`, `slope`) at bottom
    core_rows = miss_df[miss_df["Feature Set"] == "Core-10"].copy()
    full_only_rows = miss_df[miss_df["Feature Set"] == "Full-13 only"].copy()
    core_rows = core_rows.sort_values(
        ["External Pooled (%)", "Feature"], ascending=[False, True]
    )
    full_only_rows = full_only_rows.sort_values(
        ["External Pooled (%)", "Feature"], ascending=[False, True]
    )
    ordered_miss = pd.concat([core_rows, full_only_rows], ignore_index=True)

    hosp_cols = [
        "Cleveland (%)",
        "Hungary (%)",
        "Switzerland (%)",
        "Long Beach VA (%)",
    ]
    hosp_xticklabels = [
        "Cleveland\n(n=303)\n[Train]",
        "Hungary\n(n=294)\n[Ext.]",
        "Switzerland\n(n=123)\n[Ext.]",
        "Long Beach VA\n(n=200)\n[Ext.]",
    ]
    heat_data = ordered_miss[hosp_cols].to_numpy(dtype=float)
    feature_labels = ordered_miss["Feature"].tolist()
    n_core = len(core_rows)

    # Sequential colorblind-safe white -> sky -> Okabe-Ito blue -> deep navy
    cmap = LinearSegmentedColormap.from_list(
        "okabe_missing",
        ["#F8FAFC", "#D6EAF8", "#56B4E9", "#0072B2", "#003B66"],
    )

    fig, (ax_heat, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(15.0, 6.8),
        gridspec_kw={"width_ratios": [1.18, 1.05], "wspace": 0.34},
    )

    # --- Panel A: Missingness Heatmap ---
    im = ax_heat.imshow(heat_data, cmap=cmap, vmin=0.0, vmax=100.0, aspect="auto")

    # Annotate exact percentages with contrast-aware text color
    for r in range(heat_data.shape[0]):
        for c in range(heat_data.shape[1]):
            val = heat_data[r, c]
            txt_color = "white" if val >= 50.0 else "#1A1A1A"
            weight = "bold" if (r >= n_core and c > 0) or val >= 50.0 else "normal"
            ax_heat.text(
                c,
                r,
                f"{val:.1f}%",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=9.2,
                fontweight=weight,
            )

    ax_heat.set_xticks(np.arange(len(hosp_cols)))
    ax_heat.set_xticklabels(hosp_xticklabels, fontsize=9.6)
    ax_heat.set_yticks(np.arange(len(feature_labels)))
    ax_heat.set_yticklabels(feature_labels, fontsize=10.0)
    ax_heat.set_xticks(np.arange(-0.5, len(hosp_cols), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(feature_labels), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax_heat.tick_params(which="minor", bottom=False, left=False)

    # Highlight Full-13-only y-tick labels in Okabe-Ito vermilion and bold
    for idx, tick_label in enumerate(ax_heat.get_yticklabels()):
        if idx >= n_core:
            tick_label.set_color(OKABE_ITO["vermilion"])
            tick_label.set_fontweight("bold")

    # Bold divider separating Core-10 (top 10) from Full-13 only (bottom 3)
    ax_heat.axhline(
        n_core - 0.5, color=OKABE_ITO["vermilion"], linewidth=2.4, linestyle="-"
    )

    # Group labels cleanly placed in the left margin
    ax_heat.text(
        -0.25,
        (n_core - 1) / 2.0,
        "Core-10\nFeatures",
        transform=ax_heat.get_yaxis_transform(),
        ha="center",
        va="center",
        rotation=90,
        fontsize=9.6,
        fontweight="bold",
        color=OKABE_ITO["blue"],
    )
    ax_heat.text(
        -0.25,
        n_core + 1.0,
        "Full-13\nOnly",
        transform=ax_heat.get_yaxis_transform(),
        ha="center",
        va="center",
        rotation=90,
        fontsize=9.6,
        fontweight="bold",
        color=OKABE_ITO["vermilion"],
    )

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.042, pad=0.03)
    cbar.set_label("Missing Values (%)", fontsize=10.0)
    cbar.ax.tick_params(labelsize=9.2)

    ax_heat.set_title(
        "A. Feature Missingness by Hospital (Impossible Zeros Cleaned)",
        pad=12,
        loc="left",
    )

    # --- Panel B: Disease Prevalence Bar Chart ---
    hospitals = cohort_df["Hospital"].tolist()
    prevalence = cohort_df["Prevalence (%)"].to_numpy(dtype=float)
    n_counts = cohort_df["N"].to_numpy(dtype=int)
    disease_counts = cohort_df["Disease Count"].to_numpy(dtype=int)
    healthy_counts = cohort_df["Healthy Count"].to_numpy(dtype=int)

    bar_colors = [
        OKABE_ITO["blue"],
        OKABE_ITO["teal"],
        OKABE_ITO["vermilion"],
        OKABE_ITO["amber"],
    ]
    x_pos = np.arange(len(hospitals))
    bars = ax_bar.bar(
        x_pos,
        prevalence,
        width=0.56,
        color=bar_colors,
        edgecolor="#222222",
        linewidth=0.8,
        zorder=3,
    )

    # Reference line at Cleveland training prevalence
    clev_prev = float(prevalence[0])
    ax_bar.axhline(
        clev_prev,
        color=OKABE_ITO["blue"],
        linestyle="--",
        linewidth=1.3,
        alpha=0.85,
        label=f"Cleveland training prevalence ({clev_prev:.1f}%)",
        zorder=2,
    )

    # Compact 3-line annotations cleanly contained inside each bar's column width
    for bar, prev, n_tot, n_dis, n_hlth in zip(
        bars, prevalence, n_counts, disease_counts, healthy_counts
    ):
        cx = bar.get_x() + bar.get_width() / 2.0
        ax_bar.text(
            cx,
            prev + 2.0,
            f"{prev:.1f}%\nn={n_tot}\n({n_dis}/{n_hlth})",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            color="#222222",
            linespacing=1.22,
        )

    # Callout cleanly anchored in open upper-left space above Cleveland & Hungary
    ax_bar.annotate(
        "93.5% prevalence: predicting 'disease'\n"
        "for all patients yields 93.5% accuracy\n"
        "(0% specificity) \u2014 ROC-AUC &\n"
        "Balanced Accuracy are essential.",
        xy=(1.70, 90.0),
        xytext=(0.52, 84.0),
        ha="center",
        va="center",
        fontsize=8.3,
        color="#222222",
        bbox=dict(
            boxstyle="round,pad=0.42",
            facecolor="#FFF7F0",
            edgecolor=OKABE_ITO["vermilion"],
            linewidth=1.0,
        ),
        arrowprops=dict(
            arrowstyle="->",
            color=OKABE_ITO["vermilion"],
            lw=1.3,
            connectionstyle="arc3,rad=-0.12",
        ),
        zorder=5,
    )

    ax_bar.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        fontsize=8.5,
    )

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(
        ["Cleveland\n[Train]", "Hungary\n[Ext.]", "Switzerland\n[Ext.]", "Long Beach VA\n[Ext.]"],
        fontsize=9.6,
    )
    ax_bar.set_xlabel("Hospital Cohort (bar labels: prevalence, n, disease/healthy)")
    ax_bar.set_ylabel("Angiographic Heart Disease Prevalence (%)")
    ax_bar.set_ylim(0, 118)
    ax_bar.set_xlim(-0.62, 3.62)
    ax_bar.yaxis.grid(True, linestyle=":", color="#CCCCCC", zorder=0)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.set_title("B. Target Prevalence Shift Across Cohorts", pad=12, loc="left")

    fig.suptitle(
        "Cohort Audit: Cross-Hospital Missingness Structure and Disease Prevalence Shift",
        fontsize=13.5,
        fontweight="bold",
        y=1.00,
    )

    out_path = FIGURES_DIR / "fig1_cohort_audit.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig2_external_performance(metrics_df: pd.DataFrame) -> None:
    """Figure 2: Dot-whisker comparison of ROC-AUC and Balanced Accuracy (95% CI)."""
    cohort_order = [
        ("cleveland_cv", "Cleveland\n(5-fold CV, n=303)"),
        ("hungarian", "Hungary\n(n=294)"),
        ("switzerland", "Switzerland\n(n=123)"),
        ("va", "Long Beach VA\n(n=200)"),
    ]
    model_order = [
        "Full-13 + LogReg",
        "Core-10 + LogReg",
        "Full-13 + RF",
        "Core-10 + RF",
    ]
    offsets = np.array([-0.24, -0.08, 0.08, 0.24])

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.6, 6.3),
        gridspec_kw={"wspace": 0.20},
    )
    # Reserve crisp vertical separation for suptitle -> shared legend -> panel titles
    fig.subplots_adjust(top=0.79)

    panel_specs = [
        (
            axes[0],
            "roc_auc",
            "A. Discrimination: ROC-AUC (95% Bootstrap CI)",
            "ROC-AUC",
            (0.28, 1.06),
        ),
        (
            axes[1],
            "balanced_accuracy",
            "B. Thresholded Quality: Balanced Accuracy at 0.50 (95% CI)",
            "Balanced Accuracy (Threshold = 0.50)",
            (0.28, 1.06),
        ),
    ]

    x_centers = np.arange(len(cohort_order), dtype=float)

    for ax, metric, title, ylabel, ylim in panel_specs:
        # Subtle shaded background separating Internal CV from External Validation
        ax.axvspan(-0.52, 0.50, color="#F2F6FA", zorder=0)
        ax.axvline(0.50, color="#888888", linestyle=":", linewidth=1.2, zorder=1)

        # Chance reference line at 0.50
        ax.axhline(
            0.50,
            color="#555555",
            linestyle="--",
            linewidth=1.2,
            zorder=1,
        )
        ax.text(
            3.46,
            0.508,
            "Chance (0.50)",
            ha="right",
            va="bottom",
            fontsize=8.6,
            color="#555555",
            fontstyle="italic",
        )

        # Section headers cleanly inside top margin above all CIs
        ax.text(
            0.0,
            1.032,
            "Internal CV",
            ha="center",
            va="top",
            fontsize=9.2,
            fontweight="bold",
            color=OKABE_ITO["blue"],
        )
        ax.text(
            2.0,
            1.032,
            "External Validation (Frozen Cleveland Pipelines)",
            ha="center",
            va="top",
            fontsize=9.2,
            fontweight="bold",
            color="#333333",
        )

        # Plot dodged markers with asymmetric 95% bootstrap error bars
        for m_idx, model_label in enumerate(model_order):
            style = MODEL_STYLES[model_label]
            xs = x_centers + offsets[m_idx]
            ys = []
            err_low = []
            err_high = []

            for cohort_key, _ in cohort_order:
                row = metrics_df[
                    (metrics_df["cohort_key"] == cohort_key)
                    & (metrics_df["model_label"] == model_label)
                ].iloc[0]
                est = float(row[metric])
                lo = float(row[f"{metric}_ci_low"])
                hi = float(row[f"{metric}_ci_high"])
                ys.append(est)
                err_low.append(max(0.0, est - lo))
                err_high.append(max(0.0, hi - est))

            ax.errorbar(
                xs,
                ys,
                yerr=[err_low, err_high],
                fmt=str(style["marker"]),
                color=str(style["color"]),
                markerfacecolor=str(style["color"]),
                markeredgecolor="white",
                markeredgewidth=0.9,
                markersize=7.6,
                elinewidth=1.8,
                capsize=3.5,
                capthick=1.3,
                label=str(style["short"]),
                zorder=4,
            )

        ax.set_xticks(x_centers)
        ax.set_xticklabels([label for _, label in cohort_order], fontsize=9.6)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.52, 3.52)
        ax.yaxis.grid(True, linestyle=":", color="#D0D0D0", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(title, pad=10, loc="left")

    # Note regarding wide Switzerland CIs (8 healthy patients)
    axes[0].text(
        2.0,
        0.31,
        "Note: Wide Switzerland CIs reflect\nonly 8 healthy patients (n=123).",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#555555",
        fontstyle="italic",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#CCCCCC",
            alpha=0.92,
        ),
    )

    # Shared top legend positioned cleanly between suptitle and panel titles
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        fontsize=9.8,
        columnspacing=1.8,
        handletextpad=0.5,
    )

    fig.suptitle(
        "Cross-Hospital Generalization of Frozen Cleveland Models (Full-13 vs. Core-10)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )

    out_path = FIGURES_DIR / "fig2_external_performance.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig3_shift_diagnostics(
    metrics_df: pd.DataFrame,
    reliance_df: pd.DataFrame,
) -> None:
    """Figure 3: Feature reliance vs missingness (Panel A) & calibration shift (Panel B)."""
    fig, (ax_rel, ax_cal) = plt.subplots(
        1,
        2,
        figsize=(14.8, 6.4),
        gridspec_kw={"wspace": 0.24},
    )

    # --- Panel A: Reliance on Unavailable Clinical Tests (Full-13 + LogReg) ---
    # Shaded upper-right "high-reliance, high-missingness" zone
    zone_x_min = 42.0
    zone_y_min = 0.009
    zone_rect = mpatches.Rectangle(
        (zone_x_min, zone_y_min),
        106.0 - zone_x_min,
        0.072 - zone_y_min,
        facecolor="#FDECE2",
        edgecolor=OKABE_ITO["vermilion"],
        linestyle="--",
        linewidth=1.1,
        alpha=0.65,
        zorder=1,
    )
    ax_rel.add_patch(zone_rect)
    ax_rel.text(
        74.0,
        0.0695,
        "High-Reliance,\nHigh-Missingness Zone",
        ha="center",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=OKABE_ITO["vermilion"],
        zorder=2,
    )
    ax_rel.text(
        72.0,
        0.041,
        "Full-13 + RF shows same pattern:\nthal = +0.052, ca = +0.039",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#6E3514",
        fontstyle="italic",
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="#FFF9F5",
            edgecolor="#E6B89C",
            linewidth=0.8,
        ),
        zorder=2,
    )

    ax_rel.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0, zorder=1)

    for _, row in reliance_df.iterrows():
        feat = str(row["feature"])
        x_val = float(row["external_pooled_missing_pct"])
        y_val = float(row["logreg_perm_roc_auc_drop_mean"])
        is_full_only = feat in EXCLUDED_PORTABILITY_FEATURES

        color = OKABE_ITO["vermilion"] if is_full_only else OKABE_ITO["blue"]
        marker = "D" if is_full_only else "o"
        size = 95 if is_full_only else 72

        ax_rel.scatter(
            x_val,
            y_val,
            s=size,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=1.0,
            zorder=4,
        )

    # Explicit data-coordinate label targets guaranteeing clean ladder spacing
    # for the tight Core-10 cluster around 8-10% missingness and zero overlaps.
    label_targets: dict[str, tuple[float, float, str, bool]] = {
        "ca": (94.0, 0.0578, "right", False),
        "thal": (78.4, 0.0258, "center", False),
        "slope": (50.1, 0.0155, "center", False),
        "cp": (4.5, 0.0282, "left", False),
        "sex": (4.5, 0.0217, "left", False),
        "thalach": (24.0, 0.0160, "left", True),
        "trestbps": (24.0, 0.0108, "left", True),
        "oldpeak": (24.0, 0.0056, "left", True),
        "exang": (-2.0, 0.0125, "right", True),
        "restecg": (-4.5, 0.0038, "right", True),
        "chol": (32.7, -0.0045, "center", True),
        "age": (-4.5, -0.0048, "right", True),
        "fbs": (14.6, -0.0062, "center", True),
    }

    for _, row in reliance_df.iterrows():
        feat = str(row["feature"])
        x_val = float(row["external_pooled_missing_pct"])
        y_val = float(row["logreg_perm_roc_auc_drop_mean"])
        is_full_only = feat in EXCLUDED_PORTABILITY_FEATURES

        tx, ty, ha, use_arrow = label_targets[feat]
        txt = f"{feat} ({x_val:.0f}%)" if is_full_only else feat
        arrowprops = (
            dict(arrowstyle="-", color="#999999", lw=0.75) if use_arrow else None
        )
        ax_rel.annotate(
            txt,
            xy=(x_val, y_val),
            xytext=(tx, ty),
            textcoords="data",
            ha=ha,
            va="center",
            fontsize=9.1 if is_full_only else 8.4,
            fontweight="bold" if is_full_only or feat in {"cp", "sex"} else "normal",
            color=OKABE_ITO["vermilion"] if is_full_only else "#222222",
            arrowprops=arrowprops,
            zorder=5,
        )

    legend_handles_a = [
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=OKABE_ITO["vermilion"],
            markeredgecolor="white",
            markersize=8.5,
            label="Full-13 only (excluded)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=OKABE_ITO["blue"],
            markeredgecolor="white",
            markersize=8.0,
            label="Core-10 (retained)",
        ),
    ]
    ax_rel.legend(
        handles=legend_handles_a,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        fontsize=8.5,
    )

    ax_rel.set_xlim(-17.0, 108.0)
    ax_rel.set_ylim(-0.010, 0.073)
    ax_rel.set_xlabel("External Pooled Missingness Across 3 Hospitals (%)")
    ax_rel.set_ylabel("Cleveland OOF Permutation Importance\n(ROC-AUC Drop, Full-13 + LogReg)")
    ax_rel.grid(True, linestyle=":", color="#D5D5D5", zorder=0)
    ax_rel.spines["top"].set_visible(False)
    ax_rel.spines["right"].set_visible(False)
    ax_rel.set_title(
        "A. Reliance on Unavailable Clinical Tests", pad=12, loc="left"
    )

    # --- Panel B: Prevalence & Calibration Shift ---
    diag = np.linspace(16.0, 100.0, 100)
    ax_cal.plot(
        diag,
        diag,
        linestyle="--",
        color="#444444",
        linewidth=1.4,
        label="Perfect calibration (y = x)",
        zorder=2,
    )
    ax_cal.fill_between(
        diag, 16.0, diag, color="#FDECE2", alpha=0.25, zorder=0
    )

    ax_cal.text(
        99.5,
        18.2,
        "Systematic Under-prediction (Mean Pred < Obs Prev)",
        ha="right",
        va="bottom",
        fontsize=8.4,
        color=OKABE_ITO["vermilion"],
        fontweight="bold",
    )

    cal_cohorts = ["cleveland_cv", "hungarian", "va", "switzerland"]
    cal_models = [
        "Full-13 + LogReg",
        "Core-10 + LogReg",
        "Full-13 + RF",
        "Core-10 + RF",
    ]

    hosp_preds: dict[str, dict[str, float]] = {}
    hosp_obs: dict[str, float] = {}

    for model_label in cal_models:
        style = MODEL_STYLES[model_label]
        xs = []
        ys = []
        for c_key in cal_cohorts:
            row = metrics_df[
                (metrics_df["cohort_key"] == c_key)
                & (metrics_df["model_label"] == model_label)
            ].iloc[0]
            obs_pct = 100.0 * float(row["observed_prevalence"])
            pred_pct = 100.0 * float(row["mean_pred_risk"])
            xs.append(obs_pct)
            ys.append(pred_pct)
            hosp_obs[c_key] = obs_pct
            hosp_preds.setdefault(c_key, {})[model_label] = pred_pct

        order = np.argsort(xs)
        xs_sorted = np.asarray(xs)[order]
        ys_sorted = np.asarray(ys)[order]

        ax_cal.plot(
            xs_sorted,
            ys_sorted,
            color=str(style["color"]),
            linewidth=1.5,
            alpha=0.55,
            zorder=3,
        )
        ax_cal.scatter(
            xs_sorted,
            ys_sorted,
            s=76,
            color=str(style["color"]),
            marker=str(style["marker"]),
            edgecolor="white",
            linewidth=0.9,
            label=str(style["short"]),
            zorder=4,
        )

    # Compact, non-overlapping hospital callouts highlighting Full-13 vs Core-10 gaps
    cal_callouts = [
        {
            "xy": (hosp_obs["hungarian"], max(hosp_preds["hungarian"].values())),
            "xytext": (33.5, 61.0),
            "ha": "center",
            "text": "Hungary (36.1%)\nFull-13: 29%\nCore-10: 42%",
        },
        {
            "xy": (hosp_obs["cleveland_cv"], min(hosp_preds["cleveland_cv"].values())),
            "xytext": (56.5, 31.0),
            "ha": "center",
            "text": "Cleveland CV (45.9%)\nCalibrated (~46%)",
        },
        {
            "xy": (hosp_obs["va"], max(hosp_preds["va"].values())),
            "xytext": (76.0, 91.0),
            "ha": "center",
            "text": "Long Beach VA (74.5%)\nFull-13: 43% | Core-10: 66%",
        },
        {
            "xy": (hosp_obs["switzerland"], min(hosp_preds["switzerland"].values())),
            "xytext": (88.0, 35.5),
            "ha": "center",
            "text": "Switzerland (93.5%)\nFull-13: 53%\nCore-10: 65%",
        },
    ]

    for cfg in cal_callouts:
        ax_cal.annotate(
            cfg["text"],
            xy=cfg["xy"],
            xytext=cfg["xytext"],
            ha=cfg["ha"],
            va="center",
            fontsize=8.1,
            fontweight="bold",
            color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="white",
                edgecolor="#BBBBBB",
                alpha=0.94,
            ),
            arrowprops=dict(arrowstyle="->", color="#777777", lw=0.95),
            zorder=5,
        )

    ax_cal.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        fontsize=8.3,
    )

    ax_cal.set_xlim(21.0, 102.0)
    ax_cal.set_ylim(16.0, 102.0)
    ax_cal.set_xlabel("Observed Disease Prevalence (%)")
    ax_cal.set_ylabel("Mean Predicted Disease Risk (%)")
    ax_cal.grid(True, linestyle=":", color="#D5D5D5", zorder=0)
    ax_cal.spines["top"].set_visible(False)
    ax_cal.spines["right"].set_visible(False)
    ax_cal.set_title(
        "B. Prevalence & Calibration Shift Across Hospitals", pad=12, loc="left"
    )

    fig.suptitle(
        "Shift Diagnostics: Why Frozen Cleveland Models Lose Reliability Externally",
        fontsize=13.5,
        fontweight="bold",
        y=1.00,
    )

    out_path = FIGURES_DIR / "fig3_shift_diagnostics.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Main Orchestration
# ---------------------------------------------------------------------------


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    configure_matplotlib_style()

    # 1. Load frozen Cleveland models
    models = load_frozen_models()

    # 2. Load cleaned cohorts (primary protocol: impossible zeros -> NaN)
    cleveland_df = load_cohort("cleveland", clean_impossible_zeros=True)
    external_dfs = {
        key: load_cohort(key, clean_impossible_zeros=True)
        for key in EXTERNAL_HOSPITAL_KEYS
    }
    external_pooled_df = pd.concat(
        [external_dfs[key] for key in EXTERNAL_HOSPITAL_KEYS],
        ignore_index=True,
    )
    assert len(external_pooled_df) == 617, (
        f"Expected 617 pooled external patients, got {len(external_pooled_df)}"
    )

    # 3. Unified evaluation table: Cleveland CV + 3 external hospitals + External Pooled
    all_metric_rows: list[dict] = []
    all_metric_rows.extend(load_internal_cleveland_rows(cleveland_df))

    for key in EXTERNAL_HOSPITAL_KEYS:
        all_metric_rows.extend(
            evaluate_cohort(
                cohort_key=key,
                hospital_label=COHORT_FILES[key]["name"],
                evaluation_type="external_hospital",
                df=external_dfs[key],
                models=models,
            )
        )

    all_metric_rows.extend(
        evaluate_cohort(
            cohort_key="external_pooled",
            hospital_label="External Pooled",
            evaluation_type="external_pooled",
            df=external_pooled_df,
            models=models,
        )
    )

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_df.to_csv(TABLES_DIR / "external_validation_metrics.csv", index=False)
    md_text = format_external_metrics_markdown(metrics_df)
    (TABLES_DIR / "external_validation_metrics.md").write_text(md_text)

    # 4. Shift Diagnostics
    reliance_df = compute_feature_reliance_table()
    reliance_df.to_csv(
        TABLES_DIR / "shift_diagnostics_feature_reliance.csv", index=False
    )

    smd_cohorts = {
        key: (COHORT_FILES[key]["name"], external_dfs[key])
        for key in EXTERNAL_HOSPITAL_KEYS
    }
    smd_cohorts["external_pooled"] = ("External Pooled", external_pooled_df)
    smd_df = compute_covariate_smd_table(cleveland_df, smd_cohorts)
    smd_df.to_csv(TABLES_DIR / "shift_diagnostics_covariate_smd.csv", index=False)

    chol_sens_df = compute_chol_zero_sensitivity(models)
    chol_sens_df.to_csv(TABLES_DIR / "chol_zero_sensitivity.csv", index=False)

    # 5. Publication-Quality Figures (300 DPI)
    make_fig1_cohort_audit()
    make_fig2_external_performance(metrics_df)
    make_fig3_shift_diagnostics(metrics_df, reliance_df)

    # 6. Concise CLI verification output
    print("=== External Validation & Internal CV Summary ===")
    print(md_text)

    print("=== Top Covariate Shifts (|SMD| >= 0.5 vs. Cleveland) ===")
    large_smd = (
        smd_df[
            (smd_df["cohort_key"] != "external_pooled") & (smd_df["abs_smd"] >= 0.5)
        ]
        .sort_values("abs_smd", ascending=False)
        [["hospital", "feature", "cleveland_mean", "external_mean", "smd"]]
    )
    print(large_smd.to_string(index=False))

    print("\n=== Data-Hygiene Sensitivity Check (Uncleaned Impossible Zeros) ===")
    print(
        chol_sens_df[
            [
                "hospital",
                "model_label",
                "roc_auc_cleaned",
                "roc_auc_uncleaned",
                "delta_roc_auc",
                "balanced_acc_cleaned",
                "balanced_acc_uncleaned",
                "delta_brier",
                "mean_abs_prob_shift",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved tables to:  {TABLES_DIR}")
    print(f"Saved figures to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
