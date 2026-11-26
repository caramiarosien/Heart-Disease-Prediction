"""Step 1: Four-hospital cohort, missingness, impossible-zero, and covariate audit.

Loads all four UCI Heart Disease processed cohorts (both raw and cleaned),
verifies data integrity, writes audit tables to `results/tables/`, and prints
a concise verification summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    COHORT_FILES,
    CONTINUOUS_FEATURES,
    CORE_10_FEATURES,
    COUNT_FEATURES,
    FULL_13_FEATURES,
    TABLES_DIR,
    load_all_cohorts,
)

# Human-readable labels for binary/categorical covariate summaries
CATEGORY_LABELS = {
    "sex": {1: "1 (Male)"},
    "fbs": {1: "1 (FBS > 120 mg/dl)"},
    "exang": {1: "1 (Exercise angina)"},
    "cp": {
        1: "1 (Typical angina)",
        2: "2 (Atypical angina)",
        3: "3 (Non-anginal pain)",
        4: "4 (Asymptomatic)",
    },
    "restecg": {
        0: "0 (Normal)",
        1: "1 (ST-T abnormality)",
        2: "2 (LV hypertrophy)",
    },
    "slope": {1: "1 (Upsloping)", 2: "2 (Flat)", 3: "3 (Downsloping)"},
    "thal": {3: "3 (Normal)", 6: "6 (Fixed defect)", 7: "7 (Reversible defect)"},
}


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


def build_cohort_summary(cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute sample size, prevalence, baseline accuracy, demographics, and missingness."""
    rows = []
    for key, df in cohorts.items():
        n = len(df)
        disease_n = int(df["target"].sum())
        healthy_n = n - disease_n
        prevalence = 100.0 * disease_n / n
        majority_acc = max(prevalence, 100.0 - prevalence)
        rows.append(
            {
                "Hospital": COHORT_FILES[key]["name"],
                "N": n,
                "Disease Count": disease_n,
                "Prevalence (%)": round(prevalence, 1),
                "Healthy Count": healthy_n,
                "Majority-Class Baseline Accuracy (%)": round(majority_acc, 1),
                "Mean Age (SD)": f"{df['age'].mean():.1f} ({df['age'].std(ddof=1):.1f})",
                "Male (%)": round(100.0 * df["sex"].mean(), 1),
                "Any-Missing in Full-13 (%)": round(
                    100.0 * df[FULL_13_FEATURES].isna().any(axis=1).mean(), 1
                ),
                "Any-Missing in Core-10 (%)": round(
                    100.0 * df[CORE_10_FEATURES].isna().any(axis=1).mean(), 1
                ),
            }
        )
    return pd.DataFrame(rows)


def build_missingness_table(cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute per-feature missingness (%) across all four cohorts after cleaning."""
    external_df = pd.concat(
        [df for key, df in cohorts.items() if key != "cleveland"], ignore_index=True
    )
    rows = []
    for feat in FULL_13_FEATURES:
        row = {
            "Feature": feat,
            "Feature Set": "Core-10" if feat in CORE_10_FEATURES else "Full-13 only",
        }
        for key, df in cohorts.items():
            row[f"{COHORT_FILES[key]['name']} (%)"] = round(
                100.0 * df[feat].isna().mean(), 1
            )
        row["External Pooled (%)"] = round(100.0 * external_df[feat].isna().mean(), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def build_impossible_zeros_audit(
    raw_cohorts: dict[str, pd.DataFrame],
    clean_cohorts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Audit `?` markers vs biologically impossible zeros (`chol == 0`, `trestbps == 0`)."""
    rows = []
    for key in COHORT_FILES:
        raw_df = raw_cohorts[key]
        clean_df = clean_cohorts[key]
        n = len(raw_df)
        row = {"Hospital": COHORT_FILES[key]["name"], "N": n}
        for col in ["chol", "trestbps"]:
            q_n = int(raw_df[col].isna().sum())
            zero_n = int((raw_df[col] == 0).sum())
            tot_n = int(clean_df[col].isna().sum())
            assert q_n + zero_n == tot_n
            row[f"{col}_question_mark_n"] = q_n
            row[f"{col}_zero_n"] = zero_n
            row[f"{col}_zero_pct"] = round(100.0 * zero_n / n, 1)
            row[f"{col}_total_missing_n"] = tot_n
            row[f"{col}_total_missing_pct"] = round(100.0 * tot_n / n, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def build_covariate_summary(cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize clinical feature distributions across cohorts for shift diagnostics."""
    rows = []
    for key, df in cohorts.items():
        hospital = COHORT_FILES[key]["name"]

        for feat in CONTINUOUS_FEATURES + COUNT_FEATURES:
            s = df[feat].dropna()
            n_obs = int(len(s))
            q25, med, q75 = (
                (float(s.quantile(0.25)), float(s.median()), float(s.quantile(0.75)))
                if n_obs > 0
                else (np.nan, np.nan, np.nan)
            )
            rows.append(
                {
                    "Hospital": hospital,
                    "Feature": feat,
                    "Feature_Type": "continuous" if feat in CONTINUOUS_FEATURES else "count",
                    "Category": "all",
                    "N_Observed": n_obs,
                    "Mean": round(float(s.mean()), 2) if n_obs > 0 else np.nan,
                    "SD": round(float(s.std(ddof=1)), 2) if n_obs > 1 else np.nan,
                    "Median": round(med, 2) if n_obs > 0 else np.nan,
                    "Q25": round(q25, 2) if n_obs > 0 else np.nan,
                    "Q75": round(q75, 2) if n_obs > 0 else np.nan,
                    "IQR": round(q75 - q25, 2) if n_obs > 0 else np.nan,
                    "Count": pd.NA,
                    "Percent": np.nan,
                }
            )

        for feat in BINARY_FEATURES + CATEGORICAL_FEATURES:
            s = df[feat].dropna()
            n_obs = int(len(s))
            ftype = "binary" if feat in BINARY_FEATURES else "categorical"
            for code, label in CATEGORY_LABELS[feat].items():
                count = int((s == code).sum())
                pct = round(100.0 * count / n_obs, 1) if n_obs > 0 else np.nan
                rows.append(
                    {
                        "Hospital": hospital,
                        "Feature": feat,
                        "Feature_Type": ftype,
                        "Category": label,
                        "N_Observed": n_obs,
                        "Mean": np.nan,
                        "SD": np.nan,
                        "Median": np.nan,
                        "Q25": np.nan,
                        "Q75": np.nan,
                        "IQR": np.nan,
                        "Count": count,
                        "Percent": pct,
                    }
                )
    out = pd.DataFrame(rows)
    out["Count"] = out["Count"].astype("Int64")
    return out


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    raw_cohorts = load_all_cohorts(clean_impossible_zeros=False)
    clean_cohorts = load_all_cohorts(clean_impossible_zeros=True)

    cohort_summary = build_cohort_summary(clean_cohorts)
    missingness = build_missingness_table(clean_cohorts)
    zeros_audit = build_impossible_zeros_audit(raw_cohorts, clean_cohorts)
    covariates = build_covariate_summary(clean_cohorts)

    # Save CSV and Markdown outputs
    cohort_summary.to_csv(TABLES_DIR / "cohort_summary.csv", index=False)
    (TABLES_DIR / "cohort_summary.md").write_text(df_to_markdown(cohort_summary))

    missingness.to_csv(TABLES_DIR / "missingness_by_hospital.csv", index=False)
    (TABLES_DIR / "missingness_by_hospital.md").write_text(df_to_markdown(missingness))

    zeros_audit.to_csv(TABLES_DIR / "impossible_zeros_audit.csv", index=False)
    covariates.to_csv(TABLES_DIR / "covariate_summary.csv", index=False)

    # Print concise audit summary
    print("=== Cohort Summary (Cleaned Data) ===")
    print(cohort_summary.to_string(index=False))
    print("\n=== Missingness by Hospital (%, After Cleaning) ===")
    print(missingness.to_string(index=False))
    print("\n=== Biologically Impossible Zeros Audit ===")
    print(zeros_audit.to_string(index=False))
    print(f"\nSaved audit tables to: {TABLES_DIR}")


if __name__ == "__main__":
    main()
