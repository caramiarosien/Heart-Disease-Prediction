"""Reusable evaluation metrics and stratified bootstrap CIs for Heart Disease study.

Used identically by both Step 2 (`02_train_internal.py`, internal Cleveland 5-fold CV)
and Step 3 (`03_evaluate_external.py`, external validation across Hungary, Switzerland,
and Long Beach VA) so all reported numbers share a single source of truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    recall_score,
    roc_auc_score,
)

from common import RANDOM_STATE

# Canonical ordering of metrics matching MASTERPLAN.md Section 2.E
METRIC_NAMES = [
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "brier_score",
    "mean_pred_risk",
    "observed_prevalence",
    "accuracy",
    "majority_baseline_acc",
]


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute all pre-specified discrimination, threshold, and calibration metrics.

    Args:
        y_true: Binary ground-truth labels (0 = healthy, 1 = angiographic disease).
        y_prob: Predicted probabilities for the positive class (class 1).
        threshold: Fixed classification threshold (0.5 per MASTERPLAN.md).

    Returns:
        Dictionary mapping each metric name in `METRIC_NAMES` to a float value.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if y_true_arr.shape != y_prob_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_arr.shape} vs y_prob {y_prob_arr.shape}."
        )
    if np.isnan(y_prob_arr).any():
        raise ValueError("Predicted probabilities contain NaN values.")

    y_pred = (y_prob_arr >= threshold).astype(int)
    prevalence = float(np.mean(y_true_arr))

    return {
        "roc_auc": float(roc_auc_score(y_true_arr, y_prob_arr)),
        "pr_auc": float(average_precision_score(y_true_arr, y_prob_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "sensitivity": float(
            recall_score(y_true_arr, y_pred, pos_label=1, zero_division=0)
        ),
        "specificity": float(
            recall_score(y_true_arr, y_pred, pos_label=0, zero_division=0)
        ),
        "brier_score": float(brier_score_loss(y_true_arr, y_prob_arr)),
        "mean_pred_risk": float(np.mean(y_prob_arr)),
        "observed_prevalence": prevalence,
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "majority_baseline_acc": float(max(prevalence, 1.0 - prevalence)),
    }


def bootstrap_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    n_bootstrap: int = 1000,
    threshold: float = 0.5,
    ci_level: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> dict[str, dict[str, float]]:
    """Compute point estimates and stratified bootstrap percentile CIs for all metrics.

    Stratified resampling draws positives and negatives separately with replacement
    to preserve exact class counts. This ensures both classes are present in every
    replicate—critical for imbalanced cohorts such as Switzerland (8 healthy cases).

    Args:
        y_true: Binary ground-truth labels (0/1).
        y_prob: Predicted positive-class probabilities.
        n_bootstrap: Number of stratified bootstrap replicates (default: 1,000).
        threshold: Fixed classification threshold (default: 0.5).
        ci_level: Confidence interval coverage level (default: 0.95).
        random_state: Seed for reproducible resampling (default: 42).

    Returns:
        Nested dict `{metric_name: {"estimate": float, "ci_low": float, "ci_high": float}}`.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    point_estimates = compute_metrics(y_true_arr, y_prob_arr, threshold=threshold)

    pos_idx = np.flatnonzero(y_true_arr == 1)
    neg_idx = np.flatnonzero(y_true_arr == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Both classes must be present for stratified bootstrapping.")

    rng = np.random.default_rng(random_state)
    boot_samples: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}

    for _ in range(n_bootstrap):
        sampled_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        sampled_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([sampled_pos, sampled_neg])

        sample_metrics = compute_metrics(
            y_true_arr[idx], y_prob_arr[idx], threshold=threshold
        )
        for name in METRIC_NAMES:
            boot_samples[name].append(sample_metrics[name])

    alpha = (1.0 - ci_level) / 2.0
    lower_q = 100.0 * alpha
    upper_q = 100.0 * (1.0 - alpha)

    summary: dict[str, dict[str, float]] = {}
    for name in METRIC_NAMES:
        values = np.asarray(boot_samples[name], dtype=float)
        summary[name] = {
            "estimate": point_estimates[name],
            "ci_low": float(np.percentile(values, lower_q)),
            "ci_high": float(np.percentile(values, upper_q)),
        }
    return summary


def df_to_markdown(df: pd.DataFrame) -> str:
    """Format a DataFrame as a GitHub-flavored Markdown table without extra deps."""
    headers = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        cells = ["" if pd.isna(v) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
