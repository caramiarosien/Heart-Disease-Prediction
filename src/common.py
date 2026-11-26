"""Shared constants and data-loading utilities for the UCI Heart Disease study.

All schema definitions, feature subsets, and data-hygiene rules mirror the
pre-specified protocol locked in MASTERPLAN.md.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"

RANDOM_STATE = 42

# Official 14-column schema of the UCI processed heart disease files
COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

# Pre-specified feature sets (MASTERPLAN.md Section 2.B)
FULL_13_FEATURES = COLUMNS[:-1]
EXCLUDED_PORTABILITY_FEATURES = ["ca", "thal", "slope"]
CORE_10_FEATURES = [
    col for col in FULL_13_FEATURES if col not in EXCLUDED_PORTABILITY_FEATURES
]
FEATURE_SETS = {
    "full_13": FULL_13_FEATURES,
    "core_10": CORE_10_FEATURES,
}

# Explicit feature-type taxonomy for preprocessing and shift diagnostics:
# - Continuous clinical measurements + fluoroscopy vessel count (`ca`)
# - Binary indicators (`sex`, `fbs`, `exang`)
# - Multi-level nominal/ordinal categoricals (`cp`, `restecg`, `slope`, `thal`)
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
COUNT_FEATURES = ["ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]

# Numeric/binary columns passed to median imputation + StandardScaler;
# multi-level categoricals passed to mode imputation + OneHotEncoder.
NUMERIC_FEATURES = CONTINUOUS_FEATURES + COUNT_FEATURES + BINARY_FEATURES

# Sanity check: NUMERIC_FEATURES and CATEGORICAL_FEATURES strictly partition FULL_13_FEATURES
assert set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES) == set(FULL_13_FEATURES)
assert set(NUMERIC_FEATURES).isdisjoint(set(CATEGORICAL_FEATURES))

# Features where zero is biologically impossible and encodes missingness in UCI
IMPOSSIBLE_ZERO_FEATURES = ["chol", "trestbps"]

# Four independent UCI hospital cohorts
COHORT_FILES = {
    "cleveland": {
        "name": "Cleveland",
        "filename": "processed.cleveland.data",
    },
    "hungarian": {
        "name": "Hungary",
        "filename": "processed.hungarian.data",
    },
    "switzerland": {
        "name": "Switzerland",
        "filename": "processed.switzerland.data",
    },
    "va": {
        "name": "Long Beach VA",
        "filename": "processed.va.data",
    },
}


def split_feature_types(features: list[str]) -> tuple[list[str], list[str]]:
    """Split a feature list into (numeric_and_binary_cols, categorical_cols)."""
    numeric_cols = [col for col in features if col in NUMERIC_FEATURES]
    categorical_cols = [col for col in features if col in CATEGORICAL_FEATURES]
    if len(numeric_cols) + len(categorical_cols) != len(features):
        unknown = set(features) - set(FULL_13_FEATURES)
        raise ValueError(f"Unknown feature(s) requested: {sorted(unknown)}")
    return numeric_cols, categorical_cols


def get_feature_set(name: str) -> tuple[list[str], list[str], list[str]]:
    """Return (all_features, numeric_and_binary_cols, categorical_cols) for a named set."""
    if name not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set '{name}'. Expected one of {list(FEATURE_SETS)}.")
    features = list(FEATURE_SETS[name])
    numeric_cols, categorical_cols = split_feature_types(features)
    return features, numeric_cols, categorical_cols


def load_cohort(hospital_key: str, clean_impossible_zeros: bool = True) -> pd.DataFrame:
    """Load one UCI hospital cohort with standardized types and binary target.

    Args:
        hospital_key: One of 'cleveland', 'hungarian', 'switzerland', 'va'.
        clean_impossible_zeros: If True, recodes biologically impossible zeros
            (`chol == 0`, `trestbps == 0`) to `np.nan`.

    Returns:
        DataFrame with the 14 UCI columns plus binary `target` (`num > 0`).
    """
    if hospital_key not in COHORT_FILES:
        raise ValueError(
            f"Unknown hospital_key '{hospital_key}'. Expected one of {list(COHORT_FILES)}."
        )

    path = DATA_RAW_DIR / COHORT_FILES[hospital_key]["filename"]
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")

    # Enforce numeric dtypes across all predictor columns and integer outcome
    for col in FULL_13_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
    df["num"] = pd.to_numeric(df["num"], errors="raise").astype(int)

    if clean_impossible_zeros:
        for col in IMPOSSIBLE_ZERO_FEATURES:
            df.loc[df[col] == 0, col] = np.nan

    # Binary angiographic disease outcome (0 = <50% narrowing, 1 = >50% narrowing)
    df["target"] = (df["num"] > 0).astype(int)
    return df


def load_all_cohorts(clean_impossible_zeros: bool = True) -> dict[str, pd.DataFrame]:
    """Load all four UCI hospital cohorts into a dict keyed by hospital_key."""
    return {
        key: load_cohort(key, clean_impossible_zeros=clean_impossible_zeros)
        for key in COHORT_FILES
    }
