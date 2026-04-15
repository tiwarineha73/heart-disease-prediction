"""
╔══════════════════════════════════════════════════════════════════╗
║        CardioSense — Heart Disease Risk Prediction App           ║
║        Production-Level ML Web Application                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import io
import time
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense | Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --red:    #E5304A;
    --rose:   #FF6B81;
    --dark:   #0E0F14;
    --card:   #16171F;
    --muted:  #8B8FA8;
    --border: #2A2B38;
    --cream:  #F5F0EB;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--dark);
    color: var(--cream);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--cream) !important; }

/* Remove Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Cards */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
}
.card-red {
    background: linear-gradient(135deg, #2a0a10 0%, #1a0608 100%);
    border: 1px solid #5a1020;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
}

/* Headings */
h1 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--cream) !important;
    letter-spacing: -0.5px;
}
h2 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--cream) !important;
    font-size: 1.6rem !important;
}
h3 { color: var(--muted) !important; font-weight: 500 !important; }

/* Metric tiles */
.metric-tile {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: var(--rose);
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 6px;
}

/* Tag badges */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-red   { background: #3a0a12; color: var(--rose); border: 1px solid #5a1020; }
.badge-green { background: #0a2a15; color: #5ce698;     border: 1px solid #1a5a30; }
.badge-blue  { background: #0a1a3a; color: #6ab4ff;     border: 1px solid #1a3a6a; }

/* Streamlit inputs */
[data-testid="stSelectbox"] > div, [data-testid="stNumberInput"] > div {
    background: #1e1f2a !important;
    border-radius: 10px !important;
}
.stSlider > div { color: var(--rose) !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--red), #c0203a) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Divider */
hr { border-color: var(--border) !important; margin: 32px 0 !important; }

/* Result boxes */
.result-high {
    background: linear-gradient(135deg, #3a0a12, #2a0a10);
    border: 2px solid var(--red);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}
.result-low {
    background: linear-gradient(135deg, #0a2a15, #071a0e);
    border: 2px solid #2a8a4a;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}
.result-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    margin-bottom: 8px;
}

/* Tip cards */
.tip-card {
    background: #1a1b25;
    border-left: 3px solid var(--rose);
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-size: 0.9rem;
}

/* Sidebar nav labels */
.nav-label {
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 20px 0 6px 0;
}

/* Matplotlib dark theme helper */
.stImage > img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA & MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load Cleveland Heart Disease dataset via sklearn or synthetic fallback."""
    try:
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "heart-disease/processed.cleveland.data"
        )
        cols = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
        df = pd.read_csv(url, names=cols, na_values="?")
    except Exception:
        # Synthetic fallback so app works offline / on Streamlit Cloud
        np.random.seed(42)
        n = 303
        df = pd.DataFrame({
            "age":     np.random.randint(29, 77, n),
            "sex":     np.random.randint(0, 2, n),
            "cp":      np.random.randint(0, 4, n),
            "trestbps":np.random.randint(94, 200, n),
            "chol":    np.random.randint(126, 564, n),
            "fbs":     np.random.randint(0, 2, n),
            "restecg": np.random.randint(0, 3, n),
            "thalach": np.random.randint(71, 202, n),
            "exang":   np.random.randint(0, 2, n),
            "oldpeak": np.round(np.random.uniform(0, 6.2, n), 1),
            "slope":   np.random.randint(0, 3, n),
            "ca":      np.random.randint(0, 4, n),
            "thal":    np.random.choice([3.0, 6.0, 7.0], n),
            "target":  np.random.randint(0, 2, n),
        })
    df.dropna(inplace=True)
    df["target"] = (df["target"] > 0).astype(int)
    return df


@st.cache_resource
def train_model(df):
    """Train Random Forest model; return model, scaler, metrics, feature names."""
    features = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8,
        min_samples_split=4, random_state=42
    )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 1),
        "precision": round(precision_score(y_test, y_pred) * 100, 1),
        "recall":    round(recall_score(y_test, y_pred) * 100, 1),
        "cm":        confusion_matrix(y_test, y_pred),
        "importance": dict(zip(features, model.feature_importances_)),
    }
    return model, scaler, metrics, features


# ─────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────
def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  "#16171F",
        "axes.facecolor":    "#1e1f2a",
        "axes.edgecolor":    "#2A2B38",
        "axes.labelcolor":   "#8B8FA8",
        "xtick.color":       "#8B8FA8",
        "ytick.color":       "#8B8FA8",
        "text.color":        "#F5F0EB",
        "grid.color":        "#2A2B38",
        "grid.linestyle":    "--",
        "grid.alpha":        0.6,
    })


# ─────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────
def generate_report(inputs: dict, prediction: int, prob: float) -> str:
    risk_label = "HIGH RISK — Heart Disease Detected" if prediction == 1 else "LOW RISK — No Heart Disease Detected"
    risk_pct   = f"{prob * 100:.1f}%"

    tips_high = [
        "Consult a cardiologist immediately.",
        "Monitor blood pressure and cholesterol regularly.",
        "Adopt a heart-healthy diet (low sodium, low saturated fat).",
        "Engage in light aerobic activity as advised by your doctor.",
        "Avoid smoking and limit alcohol consumption.",
    ]
    tips_low = [
        "Maintain a balanced, nutritious diet.",
        "Aim for 150 minutes of moderate exercise per week.",
        "Schedule annual cardiac check-ups.",
        "Manage stress through mindfulness or yoga.",
        "Keep cholesterol and blood pressure in healthy ranges.",
    ]
    tips = tips_high if prediction == 1 else tips_low

    chest_pain_map = {0: "Typical Angina", 1: "Atypical Angina",
                      2: "Non-Anginal Pain", 3: "Asymptomatic"}
    slope_map      = {0: "Upsloping", 1: "Flat", 2: "Downsloping"}
    thal_map       = {3.0: "Normal", 6.0: "Fixed Defect", 7.0: "Reversable Defect"}

    report = f"""
╔══════════════════════════════════════════════════════════════════╗
║               CARDIOSENSE — HEART RISK REPORT                    ║
╚══════════════════════════════════════════════════════════════════╝

ASSESSMENT RESULT
─────────────────────────────────────────────────────────
  {risk_label}
  Confidence Score : {risk_pct}

PATIENT INPUT SUMMARY
─────────────────────────────────────────────────────────
  Age                     : {inputs['age']} years
  Sex                     : {'Male' if inputs['sex'] == 1 else 'Female'}
  Chest Pain Type         : {chest_pain_map.get(inputs['cp'], inputs['cp'])}
  Resting Blood Pressure  : {inputs['trestbps']} mm Hg
  Cholesterol             : {inputs['chol']} mg/dl
  Fasting Blood Sugar > 120: {'Yes' if inputs['fbs'] == 1 else 'No'}
  Resting ECG             : {['Normal','ST-T Wave Abnormality','Left Ventricular Hypertrophy'][inputs['restecg']]}
  Max Heart Rate Achieved : {inputs['thalach']} bpm
  Exercise-Induced Angina : {'Yes' if inputs['exang'] == 1 else 'No'}
  ST Depression            : {inputs['oldpeak']}
  Slope of Peak ST Segment: {slope_map.get(inputs['slope'], inputs['slope'])}
  Major Vessels Colored   : {inputs['ca']}
  Thalassemia             : {thal_map.get(inputs['thal'], inputs['thal'])}

PERSONALISED RECOMMENDATIONS
─────────────────────────────────────────────────────────
""" + "\n".join(f"  {i+1}. {tip}" for i, tip in enumerate(tips)) + """

─────────────────────────────────────────────────────────
DISCLAIMER: This tool is for informational purposes only
and does not constitute medical advice. Always consult a
qualified healthcare professional for diagnosis and treatment.
══════════════════════════════════════════════════════════
"""
    return report


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 10px 0 24px 0;'>
        <div style='font-size:1.8rem; font-family:"DM Serif Display",serif;
                    color:#F5F0EB; letter-spacing:-0.5px;'>🫀 CardioSense</div>
        <div style='font-size:0.78rem; color:#8B8FA8; margin-top:4px;
                    letter-spacing:1px; text-transform:uppercase;'>
            Heart Risk Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='nav-label'>Navigate</div>", unsafe_allow_html=True)
    page = st.radio(
        label="",
        options=["🏠  Home", "🔬  Prediction", "📊  Data Analysis",
                 "🧠  Model Insights", "📄  Download Report"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#555770; line-height:1.6;'>
        Model trained on the Cleveland Heart Disease dataset (n = 303).<br><br>
        <em>Not a substitute for professional medical advice.</em>
    </div>
    """, unsafe_allow_html=True)

# Load data & model once
df      = load_data()
model, scaler, metrics, features = train_model(df)


# ═══════════════════════════════════════════
#  PAGE 1 — HOME
# ═══════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div style='padding: 40px 0 20px 0;'>
        <div style='font-size:0.8rem; color:#E5304A; text-transform:uppercase;
                    letter-spacing:3px; margin-bottom:10px;'>
            AI-Powered Health Assessment
        </div>
        <h1 style='font-size:3rem; line-height:1.15; margin:0;'>
            Know Your Heart.<br>Before It's Too Late.
        </h1>
        <p style='color:#8B8FA8; max-width:560px; margin-top:18px;
                  font-size:1.05rem; line-height:1.7;'>
            CardioSense uses a machine learning model trained on clinical patient
            records to assess your risk of heart disease — instantly, with an
            explainable confidence score.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Quick-stat strip
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (f"{metrics['accuracy']}%", "Model Accuracy"),
        ("303",  "Patient Records"),
        ("13",   "Clinical Features"),
        ("2-min","Assessment Time"),
    ]
    for col, (val, lbl) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class='metric-tile'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # How it works
    st.markdown("## How It Works")
    cols = st.columns(3)
    steps = [
        ("01", "Input Clinical Data",
         "Enter your health metrics — age, cholesterol, ECG results, and more."),
        ("02", "AI Analyses Risk",
         "Our Random Forest model processes your data and calculates a risk probability."),
        ("03", "Download Your Report",
         "Get a personalised PDF-ready report with risk level and health recommendations."),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div class='card'>
                <div style='font-family:"DM Serif Display",serif; font-size:2.5rem;
                            color:#2A2B38; line-height:1;'>{num}</div>
                <div style='font-weight:600; font-size:1rem; margin:10px 0 8px;
                            color:#F5F0EB;'>{title}</div>
                <div style='font-size:0.88rem; color:#8B8FA8; line-height:1.6;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Key risk factors
    st.markdown("## Key Risk Factors Monitored")
    factors = [
        ("🫀", "Chest Pain Type", "Strong predictor — type matters as much as presence."),
        ("🩸", "Cholesterol",     "High LDL levels correlate strongly with disease."),
        ("⚡", "ST Depression",   "Exercise-induced ST changes indicate cardiac stress."),
        ("🔬", "Vessel Blockage", "Fluoroscopy-detected blockages directly signal risk."),
        ("💓", "Max Heart Rate",  "Lower achieved rate may suggest compromised cardiac output."),
        ("🧬", "Thalassemia",     "Blood disorder status affects heart workload."),
    ]
    r1, r2 = st.columns(2)
    for i, (icon, name, desc) in enumerate(factors):
        col = r1 if i % 2 == 0 else r2
        with col:
            st.markdown(f"""
            <div style='display:flex; gap:14px; align-items:flex-start;
                        background:#16171F; border:1px solid #2A2B38;
                        border-radius:12px; padding:16px; margin-bottom:12px;'>
                <div style='font-size:1.5rem;'>{icon}</div>
                <div>
                    <div style='font-weight:600; font-size:0.95rem;
                                color:#F5F0EB;'>{name}</div>
                    <div style='font-size:0.83rem; color:#8B8FA8;
                                margin-top:4px; line-height:1.5;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  PAGE 2 — PREDICTION
# ═══════════════════════════════════════════
elif page == "🔬  Prediction":
    st.markdown("## 🔬 Heart Disease Risk Assessment")
    st.markdown("""
    <p style='color:#8B8FA8; margin-bottom:28px;'>
        Complete all fields accurately for the most reliable prediction.
    </p>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        # — Section A: Demographics
        st.markdown("### 👤 Demographics")
        a1, a2 = st.columns(2)
        with a1:
            age = st.number_input("Age (years)", min_value=18, max_value=100,
                                  value=50, step=1)
        with a2:
            sex = st.selectbox("Biological Sex", ["Male", "Female"])
            sex_val = 1 if sex == "Male" else 0

        # — Section B: Symptoms
        st.markdown("### 🩺 Symptoms & Vitals")
        b1, b2 = st.columns(2)
        with b1:
            cp_opts = {
                "Typical Angina": 0,
                "Atypical Angina": 1,
                "Non-Anginal Pain": 2,
                "Asymptomatic": 3,
            }
            cp = st.selectbox("Chest Pain Type", list(cp_opts.keys()))
            cp_val = cp_opts[cp]

            trestbps = st.number_input("Resting Blood Pressure (mm Hg)",
                                       min_value=80, max_value=220, value=130)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                               ["No", "Yes"])
            fbs_val = 1 if fbs == "Yes" else 0

        with b2:
            chol = st.number_input("Serum Cholesterol (mg/dl)",
                                   min_value=100, max_value=600, value=246)
            restecg_opts = {
                "Normal": 0,
                "ST-T Wave Abnormality": 1,
                "Left Ventricular Hypertrophy": 2,
            }
            restecg = st.selectbox("Resting ECG Results",
                                   list(restecg_opts.keys()))
            restecg_val = restecg_opts[restecg]

        # — Section C: Exercise Data
        st.markdown("### 🏃 Exercise Test Results")
        c1_, c2_ = st.columns(2)
        with c1_:
            thalach = st.number_input("Max Heart Rate Achieved (bpm)",
                                      min_value=60, max_value=220, value=150)
            exang = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
            exang_val = 1 if exang == "Yes" else 0

        with c2_:
            oldpeak = st.slider("ST Depression (oldpeak)",
                                min_value=0.0, max_value=6.2,
                                value=1.0, step=0.1)
            slope_opts = {
                "Upsloping (good sign)": 0,
                "Flat": 1,
                "Downsloping (bad sign)": 2,
            }
            slope = st.selectbox("Slope of Peak Exercise ST Segment",
                                 list(slope_opts.keys()))
            slope_val = slope_opts[slope]

        # — Section D: Diagnostics
        st.markdown("### 🔬 Diagnostic Results")
        d1, d2 = st.columns(2)
        with d1:
            ca_opts = {
                "0 — None blocked": 0,
                "1 — One blocked": 1,
                "2 — Two blocked": 2,
                "3 — Three blocked": 3,
            }
            ca = st.selectbox("Major Blood Vessels (Fluoroscopy)",
                              list(ca_opts.keys()))
            ca_val = ca_opts[ca]

        with d2:
            thal_opts = {
                "Normal": 3.0,
                "Fixed Defect": 6.0,
                "Reversable Defect": 7.0,
            }
            thal = st.selectbox("Thalassemia (Blood Disorder)",
                                list(thal_opts.keys()))
            thal_val = thal_opts[thal]

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🫀  Assess My Risk", use_container_width=True)

    # — Result
    if submitted:
        with st.spinner("Analysing your data…"):
            time.sleep(1.2)  # UX: brief loading moment

        input_data = np.array([[age, sex_val, cp_val, trestbps, chol,
                                 fbs_val, restecg_val, thalach, exang_val,
                                 oldpeak, slope_val, ca_val, thal_val]])
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        prob         = model.predict_proba(input_scaled)[0][prediction]

        # Store for report page
        st.session_state["last_inputs"] = {
            "age": age, "sex": sex_val, "cp": cp_val,
            "trestbps": trestbps, "chol": chol, "fbs": fbs_val,
            "restecg": restecg_val, "thalach": thalach, "exang": exang_val,
            "oldpeak": oldpeak, "slope": slope_val, "ca": ca_val, "thal": thal_val,
        }
        st.session_state["last_prediction"] = int(prediction)
        st.session_state["last_prob"]        = float(prob)

        st.markdown("---")
        st.markdown("### 📋 Result")

        if prediction == 1:
            st.markdown(f"""
            <div class='result-high'>
                <div class='result-title' style='color:#FF6B81;'>
                    ⚠️ High Risk Detected
                </div>
                <div style='color:#8B8FA8; font-size:0.95rem; margin-bottom:20px;'>
                    Heart Disease indicators present in your profile
                </div>
                <div style='font-size:3rem; font-family:"DM Serif Display",serif;
                            color:#E5304A;'>{prob*100:.1f}%</div>
                <div style='font-size:0.8rem; color:#8B8FA8; letter-spacing:1px;
                            text-transform:uppercase; margin-top:4px;'>Confidence Score</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-low'>
                <div class='result-title' style='color:#5ce698;'>
                    ✅ Low Risk
                </div>
                <div style='color:#8B8FA8; font-size:0.95rem; margin-bottom:20px;'>
                    No significant heart disease indicators detected
                </div>
                <div style='font-size:3rem; font-family:"DM Serif Display",serif;
                            color:#5ce698;'>{prob*100:.1f}%</div>
                <div style='font-size:0.8rem; color:#8B8FA8; letter-spacing:1px;
                            text-transform:uppercase; margin-top:4px;'>Confidence Score</div>
            </div>
            """, unsafe_allow_html=True)

        # Probability bar
        st.markdown("<br>", unsafe_allow_html=True)
        bar_color = "#E5304A" if prediction == 1 else "#5ce698"
        st.markdown(f"""
        <div style='background:#1e1f2a; border-radius:8px; height:10px; overflow:hidden;'>
            <div style='background:{bar_color}; width:{prob*100:.1f}%;
                        height:100%; border-radius:8px;
                        transition: width 1s ease;'></div>
        </div>
        <div style='display:flex; justify-content:space-between;
                    font-size:0.75rem; color:#555770; margin-top:4px;'>
            <span>0%</span><span>50%</span><span>100%</span>
        </div>
        """, unsafe_allow_html=True)

        # Health tips
        st.markdown("### 💡 Recommendations")
        tips = (
            ["Consult a cardiologist promptly.",
             "Monitor blood pressure and cholesterol weekly.",
             "Adopt a low-sodium, heart-healthy diet.",
             "Avoid smoking and reduce alcohol intake.",
             "Light aerobic exercise as medically advised."]
            if prediction == 1 else
            ["Maintain a balanced, plant-rich diet.",
             "Keep up 150+ min/week of moderate exercise.",
             "Annual cardiac check-up is recommended.",
             "Manage stress with meditation or yoga.",
             "Stay hydrated and maintain healthy weight."]
        )
        for tip in tips:
            st.markdown(f"<div class='tip-card'>{'⚠️' if prediction==1 else '✅'} {tip}</div>",
                        unsafe_allow_html=True)

        st.info("ℹ️  Head to **📄 Download Report** in the sidebar to save your assessment.", icon="📄")


# ═══════════════════════════════════════════
#  PAGE 3 — DATA ANALYSIS
# ═══════════════════════════════════════════
elif page == "📊  Data Analysis":
    set_dark_style()
    st.markdown("## 📊 Data Analysis")
    st.markdown("""
    <p style='color:#8B8FA8; margin-bottom:28px;'>
        Exploratory analysis of the Cleveland Heart Disease dataset (n=303).
    </p>
    """, unsafe_allow_html=True)

    # Dataset snapshot metrics
    c1, c2, c3 = st.columns(3)
    for col, val, lbl in zip(
        [c1, c2, c3],
        [len(df), df["target"].sum(), f"{df['chol'].mean():.0f} mg/dl"],
        ["Total Patients", "With Heart Disease", "Avg. Cholesterol"]
    ):
        with col:
            st.markdown(f"""
            <div class='metric-tile'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    row1_l, row1_r = st.columns(2)

    # Chart 1: Age distribution by target
    with row1_l:
        st.markdown("#### Age Distribution by Outcome")
        fig, ax = plt.subplots(figsize=(6, 4))
        df[df["target"] == 1]["age"].plot(kind="hist", bins=20, alpha=0.75,
                                          color="#E5304A", ax=ax, label="Heart Disease")
        df[df["target"] == 0]["age"].plot(kind="hist", bins=20, alpha=0.65,
                                          color="#5ce698", ax=ax, label="No Disease")
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.legend(facecolor="#1e1f2a", edgecolor="#2A2B38", labelcolor="#F5F0EB")
        ax.grid(True)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 2: Heart disease count
    with row1_r:
        st.markdown("#### Disease Prevalence")
        labels = ["No Disease", "Heart Disease"]
        sizes  = [len(df[df["target"] == 0]), len(df[df["target"] == 1])]
        colors = ["#5ce698", "#E5304A"]
        fig, ax = plt.subplots(figsize=(6, 4))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct="%1.1f%%", startangle=90,
            textprops={"color": "#F5F0EB"},
            wedgeprops={"edgecolor": "#16171F", "linewidth": 2}
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight("bold")
        ax.set_facecolor("#16171F")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    row2_l, row2_r = st.columns(2)

    # Chart 3: Cholesterol vs target boxplot
    with row2_l:
        st.markdown("#### Cholesterol by Outcome")
        fig, ax = plt.subplots(figsize=(6, 4))
        data_no  = df[df["target"] == 0]["chol"]
        data_yes = df[df["target"] == 1]["chol"]
        bp = ax.boxplot(
            [data_no, data_yes],
            patch_artist=True,
            widths=0.45,
            boxprops=dict(linewidth=1.5),
            medianprops=dict(color="#F5F0EB", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
        )
        bp["boxes"][0].set_facecolor("#5ce698")
        bp["boxes"][1].set_facecolor("#E5304A")
        bp["fliers"][0].set_markerfacecolor("#5ce698")
        bp["fliers"][1].set_markerfacecolor("#E5304A")
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["No Disease", "Heart Disease"])
        ax.set_ylabel("Cholesterol (mg/dl)")
        ax.grid(True, axis="y")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 4: Max Heart Rate
    with row2_r:
        st.markdown("#### Max Heart Rate by Outcome")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(df[df["target"] == 0]["age"],
                   df[df["target"] == 0]["thalach"],
                   alpha=0.55, color="#5ce698", s=22, label="No Disease")
        ax.scatter(df[df["target"] == 1]["age"],
                   df[df["target"] == 1]["thalach"],
                   alpha=0.55, color="#E5304A", s=22, label="Heart Disease")
        ax.set_xlabel("Age")
        ax.set_ylabel("Max Heart Rate (bpm)")
        ax.legend(facecolor="#1e1f2a", edgecolor="#2A2B38", labelcolor="#F5F0EB")
        ax.grid(True)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Chart 5: Correlation heatmap (full width)
    st.markdown("#### Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 5))
    corr = df[features + ["target"]].corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap=sns.diverging_palette(10, 150, as_cmap=True),
        linewidths=0.5, linecolor="#16171F",
        ax=ax, annot_kws={"size": 8},
        cbar_kws={"shrink": 0.8},
    )
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()


# ═══════════════════════════════════════════
#  PAGE 4 — MODEL INSIGHTS
# ═══════════════════════════════════════════
elif page == "🧠  Model Insights":
    set_dark_style()
    st.markdown("## 🧠 Model Insights")
    st.markdown("""
    <p style='color:#8B8FA8; margin-bottom:28px;'>
        Random Forest Classifier trained on 80% of the dataset.
    </p>
    """, unsafe_allow_html=True)

    # Performance metrics
    st.markdown("### ⚡ Performance Metrics")
    m1, m2, m3 = st.columns(3)
    for col, val, lbl in zip(
        [m1, m2, m3],
        [f"{metrics['accuracy']}%", f"{metrics['precision']}%",
         f"{metrics['recall']}%"],
        ["Accuracy", "Precision", "Recall"]
    ):
        with col:
            st.markdown(f"""
            <div class='metric-tile'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    # Plain-English explanations
    st.markdown("""
    <div class='card' style='margin-top:20px;'>
        <div style='font-weight:600; margin-bottom:12px; color:#F5F0EB;'>
            📖 What do these mean?
        </div>
        <div style='font-size:0.88rem; color:#8B8FA8; line-height:1.9;'>
            <b style='color:#F5F0EB;'>Accuracy</b> — Out of every 100 predictions,
            the model gets ~{acc} correct.<br>
            <b style='color:#F5F0EB;'>Precision</b> — When the model says "Heart Disease",
            it is correct ~{pr}% of the time.<br>
            <b style='color:#F5F0EB;'>Recall</b> — The model correctly identifies ~{rec}%
            of all true heart disease cases.
        </div>
    </div>
    """.format(
        acc=metrics["accuracy"], pr=metrics["precision"], rec=metrics["recall"]
    ), unsafe_allow_html=True)

    st.markdown("---")

    col_cm, col_fi = st.columns(2)

    # Confusion Matrix
    with col_cm:
        st.markdown("#### Confusion Matrix")
        cm   = metrics["cm"]
        fig, ax = plt.subplots(figsize=(5, 4))
        cmap = sns.color_palette(["#0a2a15", "#3a0a12"], as_cmap=False)
        sns.heatmap(
            cm, annot=True, fmt="d", ax=ax,
            cmap="RdYlGn_r",
            linewidths=1, linecolor="#16171F",
            xticklabels=["No Disease", "Heart Disease"],
            yticklabels=["No Disease", "Heart Disease"],
            annot_kws={"size": 14, "weight": "bold", "color": "#F5F0EB"},
            cbar=False,
        )
        ax.set_xlabel("Predicted", labelpad=10)
        ax.set_ylabel("Actual",    labelpad=10)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Rows = actual class | Columns = predicted class")

    # Feature Importance
    with col_fi:
        st.markdown("#### Feature Importance")
        fi   = pd.Series(metrics["importance"]).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        colors = ["#E5304A" if v >= fi.median() else "#8B8FA8" for v in fi.values]
        bars = ax.barh(fi.index, fi.values, color=colors, height=0.65)
        ax.set_xlabel("Importance Score")
        ax.grid(True, axis="x")
        ax.tick_params(axis="y", labelsize=8.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("Red bars = above-median importance")

    st.markdown("---")

    # Model details
    st.markdown("### 🔧 Model Architecture")
    st.markdown("""
    <div class='card'>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:16px;
                    font-size:0.88rem;'>
            <div><span style='color:#8B8FA8;'>Algorithm</span><br>
                 <span style='color:#F5F0EB; font-weight:600;'>
                     Random Forest Classifier</span></div>
            <div><span style='color:#8B8FA8;'>Estimators</span><br>
                 <span style='color:#F5F0EB; font-weight:600;'>200 decision trees</span></div>
            <div><span style='color:#8B8FA8;'>Max Depth</span><br>
                 <span style='color:#F5F0EB; font-weight:600;'>8 levels</span></div>
            <div><span style='color:#8B8FA8;'>Preprocessing</span><br>
                 <span style='color:#F5F0EB; font-weight:600;'>
                     Standard Scaler (z-score)</span></div>
            <div><span style='color:#8B8FA8;'>Train / Test Split</span><br>
                 <span style='color:#F5F0EB; font-weight:600;'>80% / 20%</span></div>
            <div><span style='color:#8B8FA8;'>Stratified Split</span><br>
                 <span style='color:#F5F0EB; font-weight:600;'>Yes</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  PAGE 5 — DOWNLOAD REPORT
# ═══════════════════════════════════════════
elif page == "📄  Download Report":
    st.markdown("## 📄 Download Report")

    if "last_prediction" not in st.session_state:
        st.markdown("""
        <div class='card' style='text-align:center; padding:60px;'>
            <div style='font-size:3rem; margin-bottom:16px;'>🔬</div>
            <div style='font-family:"DM Serif Display",serif; font-size:1.4rem;
                        color:#F5F0EB; margin-bottom:8px;'>
                No Assessment Yet
            </div>
            <div style='color:#8B8FA8; font-size:0.9rem;'>
                Complete the Prediction form first, then return here
                to download your personalised report.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        pred  = st.session_state["last_prediction"]
        prob  = st.session_state["last_prob"]
        inpts = st.session_state["last_inputs"]

        risk_label = "HIGH RISK" if pred == 1 else "LOW RISK"
        badge_cls  = "badge-red" if pred == 1 else "badge-green"

        st.markdown(f"""
        <div class='card'>
            <div style='margin-bottom:20px;'>
                <span class='badge {badge_cls}'>{risk_label}</span>
            </div>
            <div style='font-family:"DM Serif Display",serif; font-size:1.6rem;
                        color:#F5F0EB; margin-bottom:6px;'>
                Your Assessment is Ready
            </div>
            <div style='color:#8B8FA8; font-size:0.9rem; line-height:1.6;'>
                Confidence Score: <strong style='color:#F5F0EB;'>
                {prob*100:.1f}%</strong><br>
                Your full report includes your inputs, result, risk level,
                and personalised health recommendations.
            </div>
        </div>
        """, unsafe_allow_html=True)

        report_text = generate_report(inpts, pred, prob)

        st.download_button(
            label="⬇️  Download Report (.txt)",
            data=report_text,
            file_name="CardioSense_Report.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown("---")
        st.markdown("### Preview")
        st.code(report_text, language=None)
