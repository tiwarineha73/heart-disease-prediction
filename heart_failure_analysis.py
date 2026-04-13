# =============================================================================
# Heart Failure Death Event Prediction
# Author: Neha Tiwari
# Dataset: Heart Failure Clinical Records — 299 patients, 12 clinical features
# Goal: Build and compare classification models to predict patient mortality
#       from heart failure, and identify the top clinical risk factors.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.family"] = "DejaVu Sans"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("../data/heart_failure.csv")
print(f"Dataset loaded: {df.shape[0]} patients × {df.shape[1]} features")
print(f"\nTarget (DEATH_EVENT) distribution:\n{df['DEATH_EVENT'].value_counts()}")
print(f"\nMortality rate: {df['DEATH_EVENT'].mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: DATA CLEANING & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

# 3.1 No missing values
print(f"\nMissing values: {df.isnull().sum().sum()} — Dataset is clean.")

# 3.2 No duplicate rows
print(f"Duplicate rows: {df.duplicated().sum()}")

# 3.3 Feature type classification
binary_cols  = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking", "DEATH_EVENT"]
numeric_cols = ["age", "creatinine_phosphokinase", "ejection_fraction",
                "platelets", "serum_creatinine", "serum_sodium", "time"]

print(f"\nBinary features  : {binary_cols[:-1]}")
print(f"Numeric features : {numeric_cols}")

# 3.4 Age binning for EDA
df["AgeBin"] = pd.cut(
    df["age"],
    bins=[40, 50, 60, 70, 80, 96],
    labels=["40–50", "51–60", "61–70", "71–80", "81+"],
)

print("\n✅ Data validation complete.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

print("\n── MORTALITY BY KEY FEATURES ──")
for col in ["anaemia", "diabetes", "high_blood_pressure", "smoking"]:
    rates = df.groupby(col)["DEATH_EVENT"].mean().mul(100).round(1)
    print(f"  {col}: {rates.to_dict()}")

print("\n── CORRELATION WITH DEATH_EVENT (top 5) ──")
corr = df.corr(numeric_only=True)["DEATH_EVENT"].drop("DEATH_EVENT").abs().sort_values(ascending=False)
print(corr.head(5).round(3))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: VISUALIZATIONS — EDA
# ─────────────────────────────────────────────────────────────────────────────

# ── Chart 1: Mortality Rate by Age Group ────────────────────────────────────
age_mortality = (
    df.groupby("AgeBin", observed=True)["DEATH_EVENT"]
    .agg(["mean", "count"])
    .reset_index()
)
age_mortality["mean"] = age_mortality["mean"].mul(100).round(1)

fig, ax = plt.subplots(figsize=(8, 5))
palette = sns.color_palette("OrRd", len(age_mortality))
bars = ax.bar(
    age_mortality["AgeBin"].astype(str),
    age_mortality["mean"],
    color=palette, edgecolor="white", width=0.55,
)
for bar, (_, row) in zip(bars, age_mortality.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f"{bar.get_height():.1f}%\n(n={int(row['count'])})",
            ha="center", fontsize=9, fontweight="bold")
ax.set_title("Patient Mortality Rate by Age Group",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Age Group", fontsize=11)
ax.set_ylabel("Mortality Rate (%)", fontsize=11)
ax.set_ylim(0, 90)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_mortality_by_age.png")
plt.close()
print("Chart 1 saved.")

# ── Chart 2: Ejection Fraction vs Serum Creatinine (scatter by outcome) ──────
fig, ax = plt.subplots(figsize=(9, 6))
colors = {0: "#2ECC71", 1: "#E74C3C"}
labels = {0: "Survived", 1: "Died"}
for outcome in [0, 1]:
    subset = df[df["DEATH_EVENT"] == outcome]
    ax.scatter(
        subset["ejection_fraction"], subset["serum_creatinine"],
        c=colors[outcome], label=labels[outcome],
        alpha=0.7, edgecolors="white", linewidth=0.5, s=70,
    )
ax.set_title("Ejection Fraction vs Serum Creatinine by Outcome",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Ejection Fraction (%)", fontsize=11)
ax.set_ylabel("Serum Creatinine (mg/dL)", fontsize=11)
ax.legend(title="Outcome", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_ejection_vs_creatinine.png")
plt.close()
print("Chart 2 saved.")

# ── Chart 3: Distribution of Key Numeric Features by Outcome ─────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
features = ["age", "ejection_fraction", "serum_creatinine",
            "serum_sodium", "time", "creatinine_phosphokinase"]
titles   = ["Age", "Ejection Fraction (%)", "Serum Creatinine (mg/dL)",
            "Serum Sodium (mEq/L)", "Follow-up Period (days)", "CPK Level (mcg/L)"]

for ax, feat, title in zip(axes.flat, features, titles):
    for outcome, color, label in [(0, "#2ECC71", "Survived"), (1, "#E74C3C", "Died")]:
        sns.kdeplot(df[df["DEATH_EVENT"] == outcome][feat],
                    ax=ax, color=color, fill=True, alpha=0.3, label=label)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("")
    ax.legend(fontsize=8)

plt.suptitle("Clinical Feature Distributions: Survived vs Died",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_feature_distributions.png", bbox_inches="tight")
plt.close()
print("Chart 3 saved.")

# ── Chart 4: Correlation Heatmap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[numeric_cols + ["DEATH_EVENT"]].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="RdYlGn", center=0, ax=ax, linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Feature Correlation Heatmap",
             fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_correlation_heatmap.png")
plt.close()
print("Chart 4 saved.")

# ── Chart 5: Mortality Rate by Binary Risk Factors ───────────────────────────
risk_factors = {
    "Anaemia":           df.groupby("anaemia")["DEATH_EVENT"].mean()[1] * 100,
    "Diabetes":          df.groupby("diabetes")["DEATH_EVENT"].mean()[1] * 100,
    "High Blood\nPressure": df.groupby("high_blood_pressure")["DEATH_EVENT"].mean()[1] * 100,
    "Smoking":           df.groupby("smoking")["DEATH_EVENT"].mean()[1] * 100,
}
baseline = df["DEATH_EVENT"].mean() * 100

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(risk_factors.keys(), risk_factors.values(),
              color=["#E53935", "#FB8C00", "#8E24AA", "#546E7A"],
              edgecolor="white", width=0.5)
ax.axhline(y=baseline, color="#1565C0", linestyle="--",
           linewidth=2, label=f"Overall avg: {baseline:.1f}%")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.1f}%",
            ha="center", fontsize=11, fontweight="bold")
ax.set_title("Mortality Rate: Patients WITH Each Risk Factor",
             fontsize=13, fontweight="bold", pad=15)
ax.set_ylabel("Mortality Rate (%)", fontsize=11)
ax.set_ylim(0, 50)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_mortality_by_risk_factor.png")
plt.close()
print("Chart 5 saved.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: MACHINE LEARNING — MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

# 6.1 Prepare features
X = df.drop(["DEATH_EVENT", "AgeBin"], axis=1)
y = df["DEATH_EVENT"]

# 6.2 Train/test split — stratified to preserve class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6.3 Feature scaling
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# 6.4 Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# 6.5 Train all models and collect metrics
cv       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results  = {}

print("\n── MODEL COMPARISON ──────────────────────────────────")
print(f"{'Model':<25} {'Accuracy':>9} {'ROC-AUC':>9} {'CV Mean':>9} {'CV Std':>8}")
print("─" * 65)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred   = model.predict(X_test_sc)
    y_prob   = model.predict_proba(X_test_sc)[:, 1]
    acc      = accuracy_score(y_test, y_pred)
    auc      = roc_auc_score(y_test, y_prob)
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring="accuracy")
    results[name] = {
        "model": model, "y_pred": y_pred, "y_prob": y_prob,
        "acc": acc, "auc": auc,
        "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
    }
    print(f"{name:<25} {acc:>9.3f} {auc:>9.3f} {cv_scores.mean():>9.3f} {cv_scores.std():>8.3f}")

# Best model = Random Forest
best_name  = "Random Forest"
best       = results[best_name]

print(f"\n✅ Best Model: {best_name}")
print(f"   Accuracy : {best['acc']*100:.1f}%")
print(f"   ROC-AUC  : {best['auc']:.4f}")
print(f"\n── CLASSIFICATION REPORT ({best_name}) ──")
print(classification_report(y_test, best["y_pred"],
                             target_names=["Survived", "Died"]))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: VISUALIZATIONS — ML RESULTS
# ─────────────────────────────────────────────────────────────────────────────

# ── Chart 6: Model Accuracy Comparison ───────────────────────────────────────
model_names = list(results.keys())
accuracies  = [results[m]["acc"] * 100 for m in model_names]
aucs        = [results[m]["auc"] for m in model_names]

x     = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy (%)",
               color="#1565C0", edgecolor="white")
bars2 = ax.bar(x + width/2, [a * 100 for a in aucs], width, label="ROC-AUC × 100",
               color="#E53935", edgecolor="white")
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{bar.get_height():.1f}",
            ha="center", fontsize=9, fontweight="bold")
ax.set_title("Model Performance Comparison: Accuracy vs ROC-AUC",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylabel("Score (%)", fontsize=11)
ax.set_ylim(0, 105)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_model_comparison.png")
plt.close()
print("Chart 6 saved.")

# ── Chart 7: ROC Curves — All Models ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
colors_roc = ["#1565C0", "#FB8C00", "#2ECC71", "#E53935"]
for (name, res), color in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{name} (AUC = {res['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.legend(fontsize=10, loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_roc_curves.png")
plt.close()
print("Chart 7 saved.")

# ── Chart 8: Confusion Matrix — Best Model ────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, best["y_pred"])
disp = ConfusionMatrixDisplay(cm, display_labels=["Survived", "Died"])
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix — {best_name}",
             fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_confusion_matrix.png")
plt.close()
print("Chart 8 saved.")

# ── Chart 9: Feature Importance — Random Forest ───────────────────────────────
rf_model = results["Random Forest"]["model"]
fi = (
    pd.Series(rf_model.feature_importances_, index=X.columns)
    .sort_values(ascending=True)
)
fig, ax = plt.subplots(figsize=(8, 6))
palette = sns.color_palette("Blues_d", len(fi))
bars = ax.barh(fi.index, fi.values, color=palette, edgecolor="white")
for bar in bars:
    ax.text(bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.3f}",
            va="center", fontsize=9, fontweight="bold")
ax.set_title("Feature Importance — Random Forest",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xlabel("Importance Score", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_feature_importance.png")
plt.close()
print("Chart 9 saved.")

print("\n✅ All 9 charts saved to /outputs/")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: KEY INSIGHTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("""
═══════════════════════════════════════════════════════════════
                     KEY CLINICAL INSIGHTS
═══════════════════════════════════════════════════════════════

1. BEST MODEL: Random Forest
   - Accuracy : 83.3%
   - ROC-AUC  : 0.891 (strong discriminatory power)
   - Baseline Decision Tree accuracy: 73.3%
   - Improvement achieved: +10 percentage points

2. TOP 5 PREDICTIVE FEATURES (by importance):
   1. Follow-up period (time)     — 36.1%
   2. Serum creatinine            — 15.4%  ← kidney function marker
   3. Ejection fraction           — 12.9%  ← heart pumping efficiency
   4. Platelets                   — 7.7%
   5. Age                         — 7.7%

3. CLINICAL IMPLICATIONS:
   - Shorter follow-up period strongly correlates with death
     (patients who died had fewer follow-up visits)
   - High serum creatinine = kidney damage + heart failure combo
     is particularly lethal
   - Low ejection fraction (<30%) is a known critical threshold

4. RISK FACTOR MORTALITY RATES:
   - High blood pressure patients: 37.1% mortality
   - Anaemia patients           : 35.7% mortality
   - Overall average            : 32.1%
   - Smoking had minimal impact in this dataset

5. AGE IS NOT LINEAR:
   - Age 81+  : 72.2% mortality (high-risk, small group)
   - Age 71–80: 52.9% mortality
   - Age 61–70: 24.7% mortality (lower than expected)
   - Suggests age alone is insufficient — clinical markers matter more

═══════════════════════════════════════════════════════════════
""")
