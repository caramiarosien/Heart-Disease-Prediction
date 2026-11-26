# Cross-Hospital External Validation of Heart Disease Prediction

Does a heart disease prediction model trained on patients from a single hospital (**Cleveland**, $n=303$) still generalize when evaluated on three independent hospitals (**Hungary** $n=294$, **Switzerland** $n=123$, **Long Beach VA** $n=200$)?

Most tutorials using the public [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) train and test exclusively on the 303 Cleveland patients and ignore the remaining 617 patients from three other clinical sites. This project implements a **pre-specified, leakage-free validation pipeline** to test cross-hospital transportability and diagnose **missingness shift**, **covariate shift**, and **prevalence/calibration shift**.

---

## Project Status

- [x] **Step 0:** Pre-specified experimental protocol locked in [`MASTERPLAN.md`](MASTERPLAN.md)
- [x] **Step 1:** Data download & four-hospital cohort/missingness audit ([`src/01_audit_data.py`](src/01_audit_data.py))
- [ ] **Step 2:** Internal 5-fold cross-validation & frozen model training on Cleveland (`src/02_train_internal.py`)
- [ ] **Step 3:** External validation across Hungary, Switzerland & Long Beach VA + shift diagnostics & figures (`src/03_evaluate_external.py`)
- [ ] **Step 4:** Concise 1–2 page technical report (`report/report.md`) & independent Devil's Advocate / QA audit

---

## Step 1 Findings: Four-Hospital Data Audit ($N = 920$)

Running `python3 src/01_audit_data.py` reveals severe **prevalence shift**, **demographic shift**, and **missingness shift** across the four UCI hospitals:

### Cohort Summary

| Hospital | N | Disease Count | Prevalence (%) | Healthy Count | Majority-Class Baseline (%) | Mean Age (SD) | Male (%) | Any-Missing Full-13 (%) | Any-Missing Core-10 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Cleveland | 303 | 139 | 45.9 | 164 | 54.1 | 54.4 (9.0) | 68.0 | 2.0 | 0.0 |
| Hungary | 294 | 106 | 36.1 | 188 | 63.9 | 47.8 (7.8) | 72.4 | 99.7 | 11.2 |
| Switzerland | 123 | 115 | 93.5 | 8 | 93.5 | 55.3 (9.0) | 91.9 | 100.0 | 100.0 |
| Long Beach VA | 200 | 149 | 74.5 | 51 | 74.5 | 59.4 (7.8) | 97.0 | 99.5 | 51.5 |

### Key Missingness & Data-Hygiene Findings

| Feature | Feature Set | Cleveland (%) | Hungary (%) | Switzerland (%) | Long Beach VA (%) | External Pooled (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ca` | Full-13 only | 1.3 | 99.0 | 95.9 | 99.0 | 98.4 |
| `thal` | Full-13 only | 0.7 | 90.5 | 42.3 | 83.0 | 78.4 |
| `slope` | Full-13 only | 0.0 | 64.6 | 13.8 | 51.0 | 50.1 |
| `chol` | Core-10 | 0.0 | 7.8 | 100.0 | 28.0 | 32.7 |
| `trestbps` | Core-10 | 0.0 | 0.3 | 1.6 | 28.5 | 9.7 |

- **Biologically impossible zeros:** `chol == 0` occurs in **123/123 (100.0%)** of Switzerland patients and **49/200 (24.5%)** of Long Beach VA patients (`trestbps == 0` in 1/200 Long Beach VA). All are recoded to `NaN` prior to Cleveland-only median imputation.
- **Why accuracy alone misleads:** Predicting "disease for everyone" achieves **93.5% accuracy** in Switzerland and **74.5%** in Long Beach VA.
- **Why Full-13 is fragile:** Nearly 100% of external patients (99.5%–100.0%) are missing at least one feature in `Full-13` (driven by `ca`, `thal`, and `slope`).

Full tables are saved in [`results/tables/`](results/tables/) and data provenance/SHA-256 checksums in [`data/raw/DATA_PROVENANCE.md`](data/raw/DATA_PROVENANCE.md).

---

## Pre-Specified Experimental Design

All modeling choices were locked in [`MASTERPLAN.md`](MASTERPLAN.md) prior to running any model:

- **Target:** Binary angiographic heart disease (`num == 0` vs. `num > 0`).
- **Data Hygiene (Zero Leakage):**
  - `?` entries and biologically impossible zeros (`chol == 0`, `trestbps == 0`) are coded as missing (`NaN`).
  - All preprocessing (median/mode imputation, standardization, one-hot encoding) is fit **strictly on Cleveland training data** inside a `scikit-learn` `Pipeline`.
- **Core Comparison (2 × 2 Design):**
  - **Feature sets:** `Full-13` (standard tutorial features) vs. `Core-10` (portable subset excluding `ca`, `thal`, and `slope`).
  - **Models:** L2-regularized Logistic Regression vs. depth-constrained Random Forest (`random_state=42`).
- **Evaluation Metrics:**
  - **Discrimination:** ROC-AUC (primary), PR-AUC, Balanced Accuracy, Sensitivity, Specificity (at fixed 0.5 threshold).
  - **Calibration:** Brier Score and predicted risk vs. observed prevalence.
  - **Context:** Accuracy reported strictly alongside the majority-class baseline.
  - **Uncertainty:** 1,000-sample stratified bootstrap 95% confidence intervals on external cohorts.

---

## Reproduction

```bash
pip install -r requirements.txt
python3 src/01_audit_data.py
```

---

## Repository Structure

```text
Heart-Disease-Prediction/
├── MASTERPLAN.md                 # Locked pre-specified protocol
├── README.md                     # Project overview and results summary
├── requirements.txt              # Minimal reproducible Python dependencies
├── data/raw/                     # UCI Heart Disease 4-hospital raw files + provenance
├── src/
│   ├── common.py                 # Shared schema, feature sets, and cohort loader
│   └── 01_audit_data.py          # Cohort, missingness, impossible-zero & covariate audit
├── results/
│   ├── models/                   # Frozen Cleveland-trained pipelines (gitignored)
│   ├── tables/                   # Reproducible CSV/Markdown audit & evaluation tables
│   └── figures/                  # Publication-ready diagnostic figures
└── report/
    └── report.md                 # 1–2 page technical summary report
```

---

## Data Source & Attribution

- **Dataset:** Heart Disease Dataset, UCI Machine Learning Repository (Janosi, Steinbrunn, Pfisterer & Detrano, 1988), CC BY 4.0.
- **Cohorts:** Cleveland Clinic Foundation, Hungarian Institute of Cardiology (Budapest), University Hospital Zurich & Basel (Switzerland), and V.A. Medical Center (Long Beach, CA).
