"""Step 2: Internal 5-fold CV on Cleveland and frozen final model training.

Implements the pre-specified modeling protocol from MASTERPLAN.md:
- Loads ONLY the Cleveland cohort (n=303) with biologically impossible zeros cleaned.
- Builds zero-leakage preprocessing + classifier pipelines (`ColumnTransformer` inside
  `Pipeline`) for the 2 x 2 design: {Full-13, Core-10} x {Logistic Regression, Random Forest}.
- Evaluates internal generalization via Stratified 5-Fold CV (`random_state=42`),
  computing per-fold mean/SD metrics, pooled out-of-fold (OOF) metrics with 1,000-sample
  stratified bootstrap 95% CIs, and out-of-fold permutation importance on raw features.
- Fits the 4 frozen final pipelines on all 303 Cleveland patients, exports `.joblib`
  artifacts to `results/models/`, and writes summary tables to `results/tables/`.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    MODELS_DIR,
    RANDOM_STATE,
    TABLES_DIR,
    get_feature_set,
    load_cohort,
)
from metrics import METRIC_NAMES, bootstrap_metrics, compute_metrics, df_to_markdown

# Pre-specified 2 x 2 experimental grid (MASTERPLAN.md Sections 2.B and 2.C)
FEATURE_SET_NAMES = ["full_13", "core_10"]
MODEL_NAMES = ["logreg", "rf"]

DISPLAY_FEATURE_SET = {
    "full_13": "Full-13",
    "core_10": "Core-10",
}
DISPLAY_MODEL = {
    "logreg": "LogReg",
    "rf": "RF",
}

N_SPLITS = 5
N_BOOTSTRAP = 1000
PERM_N_REPEATS = 30
THRESHOLD = 0.5


def build_pipeline(feature_set_name: str, model_name: str) -> Pipeline:
    """Create a zero-leakage preprocessing + estimator pipeline.

    All imputers, scalers, and one-hot encoders live inside the returned `Pipeline`
    so they are fitted strictly on training folds during cross-validation and only
    on full Cleveland when creating the frozen external-validation artifacts.

    Args:
        feature_set_name: 'full_13' or 'core_10'.
        model_name: 'logreg' or 'rf'.

    Returns:
        Unfitted scikit-learn `Pipeline` with steps `('preprocessor', 'model')`.
    """
    _, numeric_cols, categorical_cols = get_feature_set(feature_set_name)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    if model_name == "logreg":
        estimator = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    elif model_name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Expected one of {MODEL_NAMES}."
        )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def evaluate_internal_cv(
    df: pd.DataFrame,
    feature_set_name: str,
    model_name: str,
    cv: StratifiedKFold,
) -> tuple[dict[str, float | int | str], np.ndarray, pd.DataFrame]:
    """Run Stratified 5-fold CV for one (feature_set, model) combination.

    Returns:
        metrics_row: Flat dict containing pooled OOF point estimates + 95% bootstrap
            CIs and per-fold mean/SD values for all metrics.
        oof_proba: 1D array of out-of-fold positive-class probabilities (length n).
        perm_df: DataFrame of out-of-fold permutation importance (ROC-AUC drop)
            summarized by mean and SD across the 5 validation folds.
    """
    features, _, _ = get_feature_set(feature_set_name)
    X = df[features].copy()
    y = df["target"].to_numpy(dtype=int)

    oof_proba = np.full(len(df), np.nan, dtype=float)
    fold_metrics_list: list[dict[str, float]] = []
    fold_perm_means: list[np.ndarray] = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        # Fit preprocessing and estimator strictly on the training fold
        pipe = build_pipeline(feature_set_name, model_name)
        pipe.fit(X_train, y_train)

        val_proba = pipe.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = val_proba
        fold_metrics_list.append(
            compute_metrics(y_val, val_proba, threshold=THRESHOLD)
        )

        # Permute raw clinical features on the held-out validation fold
        perm = permutation_importance(
            pipe,
            X_val,
            y_val,
            scoring="roc_auc",
            n_repeats=PERM_N_REPEATS,
            random_state=RANDOM_STATE,
        )
        fold_perm_means.append(perm.importances_mean)

    if np.isnan(oof_proba).any():
        raise RuntimeError("Missing out-of-fold predictions after CV loop.")

    # Aggregate pooled OOF metrics with 1,000-sample stratified bootstrap 95% CIs
    oof_boot = bootstrap_metrics(
        y,
        oof_proba,
        n_bootstrap=N_BOOTSTRAP,
        threshold=THRESHOLD,
        random_state=RANDOM_STATE,
    )

    fold_metrics_df = pd.DataFrame(fold_metrics_list)
    metrics_row: dict[str, float | int | str] = {
        "feature_set": feature_set_name,
        "model": model_name,
        "model_label": (
            f"{DISPLAY_FEATURE_SET[feature_set_name]} + {DISPLAY_MODEL[model_name]}"
        ),
        "n_patients": int(len(df)),
        "n_folds": N_SPLITS,
        "threshold": THRESHOLD,
    }
    for metric in METRIC_NAMES:
        metrics_row[f"{metric}_oof"] = oof_boot[metric]["estimate"]
        metrics_row[f"{metric}_oof_ci_low"] = oof_boot[metric]["ci_low"]
        metrics_row[f"{metric}_oof_ci_high"] = oof_boot[metric]["ci_high"]
        metrics_row[f"{metric}_fold_mean"] = float(fold_metrics_df[metric].mean())
        metrics_row[f"{metric}_fold_sd"] = float(fold_metrics_df[metric].std(ddof=1))

    # Average validation-fold permutation importance across the 5 folds
    perm_matrix = np.vstack(fold_perm_means)  # shape: (n_folds, n_features)
    perm_df = pd.DataFrame(
        {
            "feature_set": feature_set_name,
            "model": model_name,
            "feature": features,
            "perm_roc_auc_drop_mean": perm_matrix.mean(axis=0),
            "perm_roc_auc_drop_sd": perm_matrix.std(axis=0, ddof=1),
        }
    ).sort_values("perm_roc_auc_drop_mean", ascending=False, ignore_index=True)

    return metrics_row, oof_proba, perm_df


def extract_logreg_coefficients(
    fitted_pipe: Pipeline,
    feature_set_name: str,
) -> pd.DataFrame:
    """Extract standardized/one-hot Logistic Regression coefficients and odds ratios."""
    preprocessor: ColumnTransformer = fitted_pipe.named_steps["preprocessor"]
    model: LogisticRegression = fitted_pipe.named_steps["model"]

    transformed_names = preprocessor.get_feature_names_out()
    coefs = model.coef_.ravel()
    assert len(transformed_names) == len(coefs)

    _, numeric_cols, categorical_cols = get_feature_set(feature_set_name)
    numeric_set = set(numeric_cols)

    rows = []
    for raw_out_name, coef in zip(transformed_names, coefs):
        prefix, clean_name = raw_out_name.split("__", maxsplit=1)
        if prefix == "num" and clean_name in numeric_set:
            raw_feature = clean_name
            transform = "standardized"
        else:
            # OneHotEncoder outputs `<col>_<category>`
            matched = [c for c in categorical_cols if clean_name.startswith(f"{c}_")]
            raw_feature = matched[0] if matched else clean_name
            transform = "one_hot"

        rows.append(
            {
                "feature_set": feature_set_name,
                "feature": clean_name,
                "raw_feature": raw_feature,
                "transform": transform,
                "coefficient": float(coef),
                "odds_ratio": float(np.exp(coef)),
                "abs_coefficient": float(abs(coef)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "abs_coefficient", ascending=False, ignore_index=True
    )


def format_metrics_markdown(metrics_df: pd.DataFrame) -> str:
    """Render pooled OOF (95% CI) and per-fold (mean +/- SD) Markdown tables."""

    def fmt_ci(row: pd.Series, metric: str) -> str:
        est = row[f"{metric}_oof"]
        lo = row[f"{metric}_oof_ci_low"]
        hi = row[f"{metric}_oof_ci_high"]
        return f"{est:.3f} ({lo:.3f}–{hi:.3f})"

    def fmt_mean_sd(row: pd.Series, metric: str) -> str:
        mu = row[f"{metric}_fold_mean"]
        sd = row[f"{metric}_fold_sd"]
        return f"{mu:.3f} ± {sd:.3f}"

    oof_rows = []
    fold_rows = []
    for _, row in metrics_df.iterrows():
        fset = DISPLAY_FEATURE_SET[row["feature_set"]]
        mod = DISPLAY_MODEL[row["model"]]
        oof_rows.append(
            {
                "Feature Set": fset,
                "Model": mod,
                "ROC-AUC (95% CI)": fmt_ci(row, "roc_auc"),
                "PR-AUC (95% CI)": fmt_ci(row, "pr_auc"),
                "Balanced Acc (95% CI)": fmt_ci(row, "balanced_accuracy"),
                "Sensitivity (95% CI)": fmt_ci(row, "sensitivity"),
                "Specificity (95% CI)": fmt_ci(row, "specificity"),
                "Brier Score (95% CI)": fmt_ci(row, "brier_score"),
                "Mean Pred Risk": f"{row['mean_pred_risk_oof']:.3f}",
                "Observed Prev": f"{row['observed_prevalence_oof']:.3f}",
                "Accuracy (Maj. Base)": (
                    f"{row['accuracy_oof']:.3f} ({row['majority_baseline_acc_oof']:.3f})"
                ),
            }
        )
        fold_rows.append(
            {
                "Feature Set": fset,
                "Model": mod,
                "ROC-AUC": fmt_mean_sd(row, "roc_auc"),
                "PR-AUC": fmt_mean_sd(row, "pr_auc"),
                "Balanced Acc": fmt_mean_sd(row, "balanced_accuracy"),
                "Sensitivity": fmt_mean_sd(row, "sensitivity"),
                "Specificity": fmt_mean_sd(row, "specificity"),
                "Brier Score": fmt_mean_sd(row, "brier_score"),
                "Accuracy": fmt_mean_sd(row, "accuracy"),
            }
        )

    sections = [
        "### Pooled Out-of-Fold Metrics (Cleveland, n=303, 1,000-Sample Stratified Bootstrap 95% CI)\n",
        df_to_markdown(pd.DataFrame(oof_rows)),
        "### 5-Fold Cross-Validation Summary (Mean ± SD Across Folds)\n",
        df_to_markdown(pd.DataFrame(fold_rows)),
    ]
    return "\n".join(sections)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load ONLY Cleveland (never touch external hospitals during training/CV)
    cleveland = load_cohort("cleveland", clean_impossible_zeros=True)
    assert len(cleveland) == 303, f"Expected 303 Cleveland patients, got {len(cleveland)}"

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    metrics_rows: list[dict[str, float | int | str]] = []
    perm_tables: list[pd.DataFrame] = []
    coef_tables: list[pd.DataFrame] = []

    oof_predictions = pd.DataFrame(
        {
            "patient_index": np.arange(len(cleveland), dtype=int),
            "target": cleveland["target"].to_numpy(dtype=int),
        }
    )

    y_all = cleveland["target"].to_numpy(dtype=int)

    for feature_set_name in FEATURE_SET_NAMES:
        features, _, _ = get_feature_set(feature_set_name)
        X_all = cleveland[features].copy()

        for model_name in MODEL_NAMES:
            run_key = f"{feature_set_name}_{model_name}"

            # 1. Stratified 5-fold CV + OOF permutation importance
            metrics_row, oof_proba, perm_df = evaluate_internal_cv(
                df=cleveland,
                feature_set_name=feature_set_name,
                model_name=model_name,
                cv=cv,
            )
            metrics_rows.append(metrics_row)
            perm_tables.append(perm_df)
            oof_predictions[run_key] = oof_proba

            # 2. Refit frozen final pipeline on ALL 303 Cleveland patients
            final_pipe = build_pipeline(feature_set_name, model_name)
            final_pipe.fit(X_all, y_all)

            model_path = MODELS_DIR / f"{run_key}.joblib"
            joblib.dump(final_pipe, model_path)

            # 3. Extract standardized/one-hot coefficients for Logistic Regression
            if model_name == "logreg":
                coef_tables.append(
                    extract_logreg_coefficients(final_pipe, feature_set_name)
                )

    # Write output tables
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(TABLES_DIR / "internal_cv_metrics.csv", index=False)
    (TABLES_DIR / "internal_cv_metrics.md").write_text(
        format_metrics_markdown(metrics_df)
    )

    oof_predictions.to_csv(
        TABLES_DIR / "internal_cv_oof_predictions.csv", index=False
    )

    perm_all_df = pd.concat(perm_tables, ignore_index=True)
    perm_all_df.to_csv(
        TABLES_DIR / "feature_importance_cleveland.csv", index=False
    )

    coef_all_df = pd.concat(coef_tables, ignore_index=True)
    coef_all_df.to_csv(
        TABLES_DIR / "logreg_coefficients_cleveland.csv", index=False
    )

    # Print verification summary
    print("=== Internal Cleveland 5-Fold CV Results ===")
    print(format_metrics_markdown(metrics_df))

    print("=== Top 5 Features by Out-of-Fold Permutation Importance (Full-13) ===")
    for model_name in MODEL_NAMES:
        sub = perm_all_df[
            (perm_all_df["feature_set"] == "full_13")
            & (perm_all_df["model"] == model_name)
        ].head(5)
        print(f"\nFull-13 + {DISPLAY_MODEL[model_name]}:")
        print(sub.to_string(index=False))

    print(f"\nSaved frozen Cleveland models to: {MODELS_DIR}")
    print(f"Saved internal CV tables to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
