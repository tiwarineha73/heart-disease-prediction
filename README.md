# Heart Failure Death Event Prediction

**Clinical Machine Learning — Predicting Patient Mortality from Heart Failure**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green?logo=pandas)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Project Overview

Heart failure affects over 64 million people globally. Early identification of high-risk patients can directly influence clinical interventions and save lives. This project builds and compares four machine learning classification models to predict patient mortality from heart failure — using real clinical records with 12 features per patient.

---

## Business / Clinical Problem Statement

> *"Clinicians need a data-driven model to identify which heart failure patients are at highest risk of death, based on their clinical markers — enabling prioritised monitoring, earlier intervention, and better allocation of critical care resources."*

---

## Objectives

- Identify the strongest clinical predictors of mortality in heart failure patients
- Build and compare four classification models on real patient data
- Achieve the best possible accuracy and ROC-AUC on a held-out test set
- Provide interpretable feature importance rankings for clinical use
- Deliver actionable risk stratification insights

---

## Dataset Description

| Attribute       | Detail                                                  |
|-----------------|---------------------------------------------------------|
| **Source**      | UCI Machine Learning Repository — Heart Failure Dataset |
| **Records**     | 299 patients                                            |
| **Features**    | 12 clinical features                                    |
| **Target**      | `DEATH_EVENT` (1 = died, 0 = survived)                 |
| **Mortality**   | 32.1% (96 deaths, 203 survivors)                       |
| **Missing Data**| None — complete dataset                                 |

**Features:**

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Patient age (years) |
| `anaemia` | Binary | Decrease in red blood cells (0/1) |
| `creatinine_phosphokinase` | Numeric | CPK enzyme level (mcg/L) |
| `diabetes` | Binary | Presence of diabetes (0/1) |
| `ejection_fraction` | Numeric | % of blood leaving heart per contraction |
| `high_blood_pressure` | Binary | Hypertension present (0/1) |
| `platelets` | Numeric | Platelet count (kiloplatelets/mL) |
| `serum_creatinine` | Numeric | Creatinine in blood (mg/dL) |
| `serum_sodium` | Numeric | Sodium in blood (mEq/L) |
| `sex` | Binary | 0 = Female, 1 = Male |
| `smoking` | Binary | Smoking status (0/1) |
| `time` | Numeric | Follow-up period (days) |

---

## Tools & Technologies

| Category          | Tools Used                            |
|-------------------|---------------------------------------|
| Language          | Python 3.10+                          |
| Data Processing   | Pandas, NumPy                         |
| Machine Learning  | Scikit-learn                          |
| Visualization     | Matplotlib, Seaborn                   |
| SQL Analysis      | MySQL / Standard SQL                  |
| Version Control   | Git, GitHub                           |
| Environment       | Jupyter Notebook / VS Code            |

---

## Models Used

- Logistic Regression
- Decision Tree
- **Random Forest** ← Best performing
- Gradient Boosting

---

## Model Results

| Model | Accuracy | ROC-AUC | CV Mean |
|-------|----------|---------|---------|
| Logistic Regression | 81.7% | 0.859 | 83.3% |
| Decision Tree | 73.3% | 0.664 | 80.0% |
| **Random Forest** | **83.3%** | **0.891** | **84.6%** |
| Gradient Boosting | 83.3% | 0.845 | 81.6% |

**Best Model: Random Forest**
- Accuracy: **83.3%**
- ROC-AUC: **0.891**
- 5-fold CV Mean: **84.6%**

---

## Key Insights

| # | Finding | Significance |
|---|---------|--------------|
| 1 | **Follow-up time** is the top predictor (36.1% importance) | Patients who died had fewer follow-up visits |
| 2 | **Serum creatinine** is 2nd most important (15.4%) | Kidney damage + heart failure is highly lethal |
| 3 | **Ejection fraction** is 3rd (12.9%) | Low EF (<30%) indicates critically weak heart function |
| 4 | Age 81+ has **72.2% mortality** vs 24.7% for 61–70 | Age group matters — but isn't the sole driver |
| 5 | **High BP patients: 37.1% mortality** vs 29.4% without | Hypertension meaningfully raises risk |
| 6 | **Smoking had minimal impact** in this cohort | Counter-intuitive finding worth flagging |

---

## Visualizations Generated

| File | Description |
|------|-------------|
| `01_mortality_by_age.png` | Mortality rate by age group with sample sizes |
| `02_ejection_vs_creatinine.png` | Scatter: key clinical markers by outcome |
| `03_feature_distributions.png` | KDE distributions: survived vs died |
| `04_correlation_heatmap.png` | Feature correlation matrix |
| `05_mortality_by_risk_factor.png` | Mortality rate for each binary risk factor |
| `06_model_comparison.png` | Accuracy and ROC-AUC comparison across all models |
| `07_roc_curves.png` | ROC curves for all 4 models |
| `08_confusion_matrix.png` | Best model confusion matrix |
| `09_feature_importance.png` | Random Forest feature importance ranking |

---

## Project Structure

```
heart-disease-prediction/
│
├── data/
│   └── heart_failure.csv               # Clinical dataset (299 patients)
│
├── notebooks/
│   └── heart_failure_analysis.py       # Full EDA + ML pipeline script
│
├── src/
│   └── heart_failure_queries.sql       # 10 clinical SQL queries
│
├── outputs/
│   ├── 01_mortality_by_age.png
│   ├── 02_ejection_vs_creatinine.png
│   ├── 03_feature_distributions.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_mortality_by_risk_factor.png
│   ├── 06_model_comparison.png
│   ├── 07_roc_curves.png
│   ├── 08_confusion_matrix.png
│   └── 09_feature_importance.png
│
├── README.md
└── requirements.txt
```

---

## How to Run

**Step 1 — Clone the repository**
```bash
git clone https://github.com/tiwarineha73/heart-disease-prediction.git
cd heart-disease-prediction
```

**Step 2 — Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — Run the analysis**
```bash
cd notebooks
python heart_failure_analysis.py
```

**Step 5 — View outputs**
All 9 charts are saved automatically to the `/outputs/` folder.

---

## ⚠️ Resume Accuracy Note

This dataset contains **299 patients and 12 features** — the UCI Heart Failure Clinical Records dataset. Resume claims should reflect these actual numbers.

---

## Author

**Neha Tiwari**
Data Analyst | Python • SQL • Machine Learning • Power BI

- GitHub: [github.com/tiwarineha73](https://github.com/tiwarineha73)
- LinkedIn: [linkedin.com/in/neha-tiwari](https://linkedin.com/in/neha-tiwari)
- Email: tiwari.neha3111@gmail.com

---

*This project is part of an end-to-end data analytics and machine learning portfolio built on real clinical datasets.*

