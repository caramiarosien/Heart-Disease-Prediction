# Cross-Hospital External Validation of Heart Disease Prediction

Does a heart disease prediction model trained on patients from a single hospital (**Cleveland**, $n=303$) still generalize when evaluated on three independent hospitals (**Hungary** $n=294$, **Switzerland** $n=123$, **Long Beach VA** $n=200$)?

Most tutorials using the public [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) train and test exclusively on the 303 Cleveland patients and ignore the remaining 617 patients from three other clinical sites. This project implements a **pre-specified, leakage-free validation pipeline** to test cross-hospital transportability and diagnose **missingness shift**, **covariate shift**, and **prevalence/calibration shift**.

---

## Project Status

- [x] **Step 0:** Pre-specified experimental protocol locked in [`MASTERPLAN.md`](MASTERPLAN.md)
- [ ] **Step 1:** Data download & four-hospital cohort/missingness audit (`src/01_audit_data.py`)
- [ ] **Step 2:** Internal 5-fold cross-validation & frozen model training on Cleveland (`src/02_train_internal.py`)
- [ ] **Step 3:** External validation across Hungary, Switzerland & Long Beach VA + shift diagnostics & figures (`src/03_evaluate_external.py`)
- [ ] **Step 4:** Concise 1–2 page technical report (`report/report.md`) & full end-to-end verification

---

## Pre-Specified Experimental Design

All modeling choices were locked in [`MASTERPLAN.md`](MASTERPLAN.md) prior to running any model:

- **Target:** Binary angiographic heart disease (`num == 0` vs. `num > 0`).
- **Data Hygiene (Zero Leakage):**
  - `?` entries and biologically impossible zeros (`chol == 0`, `trestbps == 0`) are coded as missing (`NaN`).
  - All preprocessing (median/mode imputation, standardization, one-hot encoding) is fit **strictly on Cleveland training data** inside a `scikit-learn` `Pipeline`.
- **Core Comparison (2 × 2 Design):**
  - **Feature sets:** `Full-13` (standard tutorial features) vs. `Core-10` (portable subset excluding `ca`, `thal`, and `slope`, which are heavily missing outside Cleveland).
  - **Models:** L2-regularized Logistic Regression vs. depth-constrained Random Forest (`random_state=42`).
- **Evaluation Metrics:**
  - **Discrimination:** ROC-AUC (primary), PR-AUC, Balanced Accuracy, Sensitivity, Specificity (at fixed 0.5 threshold).
  - **Calibration:** Brier Score and predicted risk vs. observed prevalence.
  - **Context:** Accuracy reported strictly alongside the majority-class baseline (crucial under prevalence shifts up to ~93%).
  - **Uncertainty:** 1,000-sample stratified bootstrap 95% confidence intervals on external cohorts.

---

## Repository Structure

```text
Heart-Disease-Prediction/
├── MASTERPLAN.md                 # Locked pre-specified protocol
├── README.md                     # Project overview and results summary
├── requirements.txt              # Minimal reproducible Python dependencies
├── data/raw/                     # UCI Heart Disease 4-hospital raw files
├── src/                          # Modular, readable analysis scripts (01 → 03)
├── results/
│   ├── models/                   # Frozen Cleveland-trained pipelines
│   ├── tables/                   # Audit & evaluation tables
│   └── figures/                  # Publication-ready diagnostic figures
└── report/
    └── report.md                 # 1–2 page technical summary report
```

---

## Data Source & Attribution

- **Dataset:** Heart Disease Dataset, UCI Machine Learning Repository (Janosi, Steinbrunn, Pfisterer & Detrano, 1988), CC BY 4.0.
- **Cohorts:** Cleveland Clinic Foundation, Hungarian Institute of Cardiology (Budapest), University Hospital Zurich & Basel (Switzerland), and V.A. Medical Center (Long Beach, CA).
