# UCI Heart Disease Dataset — Data Provenance

## Source & Attribution

- **Dataset:** Heart Disease Dataset (4-Hospital Processed Cohort Files)
- **Repository:** UCI Machine Learning Repository — <https://archive.ics.uci.edu/dataset/45/heart+disease>
- **DOI:** `10.24432/C52P4X`
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Download Date:** 2026-11-26
- **Direct Source URLs:**
  - `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data`
  - `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data`
  - `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data`
  - `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data`

### Citation

> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

> Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J.-J., Sandhu, S., Guppy, K. H., Lee, S., & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*, 64(5), 304–310.

---

## Verified File Manifest

All four files are headerless comma-separated text files with **14 columns per row** (total **920 patients** across four independent clinical centers).

| Hospital Cohort | Clinical Center | Filename | Rows | Columns | SHA-256 Checksum |
| --- | --- | --- | ---: | ---: | --- |
| Cleveland | Cleveland Clinic Foundation (Cleveland, OH, USA) | `processed.cleveland.data` | 303 | 14 | `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8` |
| Hungary | Hungarian Institute of Cardiology (Budapest, Hungary) | `processed.hungarian.data` | 294 | 14 | `d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796` |
| Switzerland | University Hospital, Zurich & Basel (Switzerland) | `processed.switzerland.data` | 123 | 14 | `834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314` |
| Long Beach VA | V.A. Medical Center (Long Beach, CA, USA) | `processed.va.data` | 200 | 14 | `e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f` |

---

## Column Definitions (14 Processed UCI Attributes)

| # | Column | Type | Clinical Definition & Coding | Feature Set |
| ---: | --- | --- | --- | --- |
| 1 | `age` | Continuous | Age in years | Core-10 & Full-13 |
| 2 | `sex` | Binary | Biological sex (`1` = male, `0` = female) | Core-10 & Full-13 |
| 3 | `cp` | Categorical | Chest pain type (`1` = typical angina, `2` = atypical angina, `3` = non-anginal pain, `4` = asymptomatic) | Core-10 & Full-13 |
| 4 | `trestbps` | Continuous | Resting blood pressure on admission (mm Hg) | Core-10 & Full-13 |
| 5 | `chol` | Continuous | Serum cholesterol (mg/dl) | Core-10 & Full-13 |
| 6 | `fbs` | Binary | Fasting blood sugar > 120 mg/dl (`1` = true, `0` = false) | Core-10 & Full-13 |
| 7 | `restecg` | Categorical | Resting ECG results (`0` = normal, `1` = ST-T wave abnormality, `2` = probable/definite LV hypertrophy) | Core-10 & Full-13 |
| 8 | `thalach` | Continuous | Maximum heart rate achieved during exercise (bpm) | Core-10 & Full-13 |
| 9 | `exang` | Binary | Exercise-induced angina (`1` = yes, `0` = no) | Core-10 & Full-13 |
| 10 | `oldpeak` | Continuous | ST depression induced by exercise relative to rest (mm) | Core-10 & Full-13 |
| 11 | `slope` | Categorical | Slope of peak exercise ST segment (`1` = upsloping, `2` = flat, `3` = downsloping) | Full-13 only |
| 12 | `ca` | Count / Numeric | Number of major vessels (`0`–`3`) colored by fluoroscopy | Full-13 only |
| 13 | `thal` | Categorical | Thallium stress test result (`3` = normal, `6` = fixed defect, `7` = reversible defect) | Full-13 only |
| 14 | `num` | Outcome | Angiographic diagnosis of heart disease (`0` = <50% diameter narrowing; `1`–`4` = >50% narrowing). Binarized in this study to `target = (num > 0)`. | Target |

---

## Known Data-Hygiene Notes

1. **Explicit missing marker:** Missing values are encoded as `?` in the raw files.
2. **Biologically impossible zeros:** In the Switzerland (`chol == 0` for all 123 patients) and Long Beach VA (`chol == 0` for 49 patients; `trestbps == 0` for 1 patient) cohorts, missing measurements were recorded as `0`. Per `MASTERPLAN.md`, `chol == 0` and `trestbps == 0` are recoded to `NaN` during preprocessing and tracked explicitly in `results/tables/impossible_zeros_audit.csv`.
3. **Raw files are immutable:** Files in `data/raw/` are kept unmodified as downloaded from UCI; all cleaning occurs programmatically in `src/common.py`.
