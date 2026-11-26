# Masterplan: Cross-Hospital External Validation of Heart Disease Prediction

> **Status:** Pre-specified protocol locked before model training (`2026-11-26`).  
> **Ziel:** Wissenschaftlich saubere externe Validierung eines klinischen Tabellendaten-Modells über vier unabhängige Kliniken des UCI Heart Disease Dataset.

---

## 1. Forschungsfrage & Kernidee

**Leitfrage:** *Generalisiert ein klassisches Herzkrankheits-Vorhersagemodell, das auf Patientendaten einer Klinik (Cleveland, $n=303$) trainiert wurde, auf drei unabhängige Kliniken (Ungarn $n=294$, Schweiz $n=123$, Long Beach VA $n=200$) — und welche Daten- und Klinikunterschiede erklären etwaige Leistungs- oder Kalibrierungsänderungen?*

**Warum dieser Ansatz:**
- Die meisten Tutorials trainieren und testen ausschließlich auf Cleveland ($n=303$) mit allen 13 Merkmalen.
- Der öffentliche UCI-Datensatz enthält drei weitere Kliniken, die fast überall ignoriert werden — und in denen mehrere zentrale Cleveland-Merkmale kaum erhoben wurden oder klinische Verteilungen stark abweichen.
- Der Vergleich zwischen einem **Tutorial-Modell (`Full-13`)** und einem **klinik-portablen Modell (`Core-10`)** über alle vier Standorte prüft externe Validität, Robustheit und Kalibrierung unter realem Hospital Shift.

---

## 2. Vorab fixiertes Versuchsprotokoll (Pre-Specification)

Alle folgenden Entscheidungen sind **vor dem ersten Modelllauf eingefroren** (kein Post-hoc-Tuning an externen Testdaten, kein Metrik-Shopping):

### A. Daten & Datenhygiene
- **Quelle:** UCI Machine Learning Repository — Heart Disease Dataset (Janosi, Steinbrunn, Pfisterer & Detrano, 1988; 4 Dateien: `processed.cleveland.data`, `processed.hungarian.data`, `processed.switzerland.data`, `processed.va.data`).
- **Zielgröße (`target`):** Binarisierung von `num`:
  - `0` = keine relevante Gefäßverengung (`num == 0`)
  - `1` = angiografisch bestätigte Herzkrankheit (`num in {1, 2, 3, 4}`)
- **Fehlende & unmögliche Werte:**
  - `?` → `NaN`
  - **Biologisch unmögliche Nullen:** `chol == 0` und `trestbps == 0` → `NaN` (z. B. kodiert die Schweizer Klinik fehlendes Serumcholesterin bei allen 123 Patienten als `0`).
- **Zero-Leakage-Prinzip:** Imputation (Median für numerische, Modus für kategoriale Merkmale), Standardisierung (`StandardScaler`) und One-Hot-Encoding (`OneHotEncoder(handle_unknown="ignore")`) werden in einer `scikit-learn`-`Pipeline` **ausschließlich auf den jeweiligen Cleveland-Trainingsdaten** gefittet.

### B. Feature-Sets (Kernvergleich)
1. **`Full-13` (Tutorial-/Standard-Setup):** Alle 13 klassischen Merkmale:
   `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.
2. **`Core-10` (Klinik-portables Setup):** Die 10 breiter erhobenen klinischen Merkmale — ohne die drei außerhalb Clevelands überwiegend fehlenden Merkmale `ca`, `thal` und `slope`.

### C. Modellfamilien (`random_state = 42`)
Beide Feature-Sets werden mit je zwei vorab fixierten, tabellendaten-geeigneten Modellen trainiert (2 × 2-Design):
1. **Logistic Regression** (`LogisticRegression(C=1.0, max_iter=1000, random_state=42)`) — interpretierbarer klinischer Standard, gut kalibrierte Wahrscheinlichkeiten.
2. **Random Forest** (`RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42)`) — nichtlineares Vergleichsmodell zur Prüfung von Overfitting vs. Generalisierung.

### D. Validierungsdesign
1. **Interne Validierung (Cleveland, $n=303$):** Stratifizierte 5-fach Kreuzvalidierung (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`), sodass jeder Cleveland-Patient genau einmal out-of-fold bewertet wird.
2. **Externe Validierung (Ungarn, Schweiz, Long Beach VA):** Refit der identischen Pipelines auf Gesamt-Cleveland ($n=303$), anschließend unveränderte Evaluation (ohne Nachtraining und ohne Schwellenwert-Anpassung, fester Schwellenwert $0.5$) auf den drei externen Kliniken.
3. **Konfidenzintervalle:** 1.000-faches stratifiziertes Bootstrap-Resampling (95 %-Perzentil-CI) für alle externen Kennzahlen.

### E. Kennzahlen
- **Primär (Diskriminierung, prävalenz-unabhängig):** `ROC-AUC`
- **Sekundär (Entscheidungsqualität am 0.5-Schwellenwert):** `Balanced Accuracy`, `Sensitivity (Recall)`, `Specificity`, `PR-AUC`
- **Kalibrierung:** `Brier Score` sowie Vergleich *mittlere vorhergesagte Wahrscheinlichkeit vs. beobachtete Krankheitsprävalenz*
- **Kontext:** `Accuracy` ausschließlich neben der **Majority-Class-Baseline**, um den Prävalenz-Effekt (insb. Schweiz ~93 % krank) sichtbar zu machen.

---

## 3. Ursachenanalyse (Shift-Diagnostik)

1. **Missingness Shift:** Fehlquoten je Merkmal und Klinik (`?` und biologisch unmögliche `0`-Werte getrennt dokumentiert).
2. **Covariate Shift:** Verteilungsunterschiede bei Alter, Geschlecht, Brustschmerztyp (`cp`), Ruheblutdruck, Cholesterin, maximaler Herzfrequenz (`thalach`) und `oldpeak`.
3. **Prevalence & Calibration Shift:** Krankheitsanteil (36 %–93 %) vs. durchschnittlich vorhergesagtes Risiko der Cleveland-Modelle.
4. **Feature-Abhängigkeit:** Vergleich der wichtigsten Prädiktoren in Cleveland mit deren externer Verfügbarkeit.

---

## 4. Repository-Struktur & Lieferumfang

```text
Heart-Disease-Prediction/
├── MASTERPLAN.md
├── README.md
├── requirements.txt
├── data/
│   └── raw/                          # 4 UCI-Klinikdateien + Quellenhinweis
├── src/
│   ├── 01_audit_data.py              # Daten-Download/Audit, Missingness- & Kohortentabellen
│   ├── 02_train_internal.py          # 5-fold CV auf Cleveland, Feature-Wichtigkeit, finale Pipelines
│   └── 03_evaluate_external.py       # Externe Evaluation (Bootstrap-CIs), Shift-Diagnostik, 3 Grafiken
├── results/
│   ├── models/                       # Eingefrorene Cleveland-Pipelines (.joblib)
│   ├── tables/                       # Reproduzierbare CSV/JSON/MD-Ergebnistabellen
│   └── figures/
│       ├── fig1_cohort_audit.png
│       ├── fig2_external_performance.png
│       └── fig3_shift_diagnostics.png
└── report/
    └── report.md                     # 1–2 Seiten englischer Ergebnisbericht
```

---

## 5. Umsetzungsplan

| Schritt | Inhalt | Status |
| --- | --- | --- |
| **Schritt 0** | Protokoll als `MASTERPLAN.md` einfrieren, `README.md` + `requirements.txt` anlegen, auf GitHub pushen | In Arbeit |
| **Schritt 1** | UCI-Rohdaten laden & `src/01_audit_data.py` (Kohorten- & Missingness-Audit, `?` und `0`-Analyse) | Offen |
| **Schritt 2** | `src/02_train_internal.py` (5-fold CV auf Cleveland für `Full-13` & `Core-10` × `LogReg` & `RF`, Feature-Wichtigkeit, finale Modelle speichern) | Offen |
| **Schritt 3** | `src/03_evaluate_external.py` (externe Validierung mit 95 %-Bootstrap-CIs, Shift-Diagnostik, 3 publikationsreife Grafiken) | Offen |
| **Schritt 4** | Englischer 1–2-Seiten-Bericht (`report/report.md`), finale `README.md`-Aktualisierung & End-to-End-Verifikation | Offen |
