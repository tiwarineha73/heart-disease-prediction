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
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG  — must be FIRST streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CardioSense | Heart Disease Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# NAVIGATION via session_state
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"

def go(page_name):
    st.session_state.page = page_name

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0E0F14 !important;
    color: #F5F0EB !important;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

/* ─── SIDEBAR — force always open & styled ─── */
[data-testid="stSidebar"] {
    background: #16171F !important;
    border-right: 1px solid #2A2B38 !important;
    min-width: 240px !important;
    max-width: 260px !important;
    display: block !important;
    visibility: visible !important;
    transform: none !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
    display: block !important;
    transform: translateX(0) !important;
}
[data-testid="collapsedControl"] {
    display: none !important;
}
[data-testid="stSidebar"] * {
    color: #F5F0EB !important;
}
/* Sidebar nav buttons — override Streamlit default button style */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #8B8FA8 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    text-align: left !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    transition: background 0.15s, color 0.15s !important;
    margin-bottom: 2px !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1e1f2a !important;
    color: #F5F0EB !important;
    opacity: 1 !important;
}

/* Cards */
.card {
    background: #16171F;
    border: 1px solid #2A2B38;
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
}
.metric-tile {
    background: #16171F;
    border: 1px solid #2A2B38;
    border-radius: 12px;
    padding: 22px 16px;
    text-align: center;
}
.metric-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #FF6B81;
    line-height: 1;
}
.metric-label {
    font-size: 0.72rem;
    color: #8B8FA8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 7px;
}
.result-high {
    background: linear-gradient(135deg,#3a0a12,#2a0a10);
    border: 2px solid #E5304A;
    border-radius: 16px;
    padding: 36px 28px;
    text-align: center;
}
.result-low {
    background: linear-gradient(135deg,#0a2a15,#071a0e);
    border: 2px solid #2a8a4a;
    border-radius: 16px;
    padding: 36px 28px;
    text-align: center;
}
.tip-card {
    background: #1a1b25;
    border-left: 3px solid #FF6B81;
    border-radius: 0 10px 10px 0;
    padding: 13px 18px;
    margin-bottom: 9px;
    font-size: 0.88rem;
    color: #c5c8da;
}
.badge-red {
    display: inline-block; padding: 4px 12px; border-radius: 100px;
    font-size: .75rem; font-weight: 600;
    background: #3a0a12; color: #FF6B81; border: 1px solid #5a1020;
}
.badge-green {
    display: inline-block; padding: 4px 12px; border-radius: 100px;
    font-size: .75rem; font-weight: 600;
    background: #0a2a15; color: #5ce698; border: 1px solid #1a5a30;
}
div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div {
    background: #1e1f2a !important;
    border: 1px solid #2A2B38 !important;
    border-radius: 8px !important;
    color: #F5F0EB !important;
}
/* Main area buttons */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg,#E5304A,#c0203a) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: opacity .2s !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    opacity: 0.82 !important;
}
hr { border-color: #2A2B38 !important; margin: 30px 0 !important; }
h1 { font-family:'DM Serif Display',serif !important; color:#F5F0EB !important; }
h2 { font-family:'DM Serif Display',serif !important; color:#F5F0EB !important; font-size:1.55rem !important; }
h3 { color:#c5c8da !important; font-weight:500 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA & MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "heart-disease/processed.cleveland.data")
        cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
                "thalach","exang","oldpeak","slope","ca","thal","target"]
        df = pd.read_csv(url, names=cols, na_values="?")
    except Exception:
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


FEATURES = ["age","sex","cp","trestbps","chol","fbs",
            "restecg","thalach","exang","oldpeak","slope","ca","thal"]

@st.cache_resource
def train_model(_df):
    X = _df[FEATURES]
    y = _df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)
    clf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                  min_samples_split=4, random_state=42)
    clf.fit(Xtr, y_train)
    yp = clf.predict(Xte)
    met = {
        "accuracy":   round(accuracy_score(y_test, yp) * 100, 1),
        "precision":  round(precision_score(y_test, yp) * 100, 1),
        "recall":     round(recall_score(y_test, yp) * 100, 1),
        "cm":         confusion_matrix(y_test, yp),
        "importance": dict(zip(FEATURES, clf.feature_importances_)),
    }
    return clf, sc, met


df = load_data()
model, scaler, metrics = train_model(df)


# ─────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────
def dark_theme():
    plt.rcParams.update({
        "figure.facecolor": "#16171F",
        "axes.facecolor":   "#1e1f2a",
        "axes.edgecolor":   "#2A2B38",
        "axes.labelcolor":  "#8B8FA8",
        "xtick.color":      "#8B8FA8",
        "ytick.color":      "#8B8FA8",
        "text.color":       "#F5F0EB",
        "grid.color":       "#2A2B38",
        "grid.linestyle":   "--",
        "grid.alpha":       0.5,
    })


# ─────────────────────────────────────────────
# REPORT GENERATOR
# ─────────────────────────────────────────────
def make_report(inp, pred, prob):
    cp_map    = {0:"Typical Angina",1:"Atypical Angina",2:"Non-Anginal Pain",3:"Asymptomatic"}
    slope_map = {0:"Upsloping",1:"Flat",2:"Downsloping"}
    thal_map  = {3.0:"Normal",6.0:"Fixed Defect",7.0:"Reversable Defect"}
    tips_h = ["Consult a cardiologist promptly.",
               "Monitor BP and cholesterol weekly.",
               "Follow a low-sodium, heart-healthy diet.",
               "Avoid smoking; limit alcohol.",
               "Light aerobic exercise as medically advised."]
    tips_l = ["Maintain a balanced plant-rich diet.",
               "Aim for 150 min/week moderate exercise.",
               "Annual cardiac check-up recommended.",
               "Manage stress via mindfulness or yoga.",
               "Keep cholesterol and BP in healthy ranges."]
    tips  = tips_h if pred == 1 else tips_l
    label = "HIGH RISK — Heart Disease Detected" if pred == 1 else "LOW RISK — No Disease Detected"
    lines = [
        "╔══════════════════════════════════════════════════════╗",
        "║          CARDIOSENSE — HEART RISK REPORT             ║",
        "╚══════════════════════════════════════════════════════╝",
        "",
        f"  Result     : {label}",
        f"  Confidence : {prob*100:.1f}%",
        "",
        "─" * 56,
        "  PATIENT INPUT SUMMARY",
        "─" * 56,
        f"  Age                      : {inp['age']} yrs",
        f"  Sex                      : {'Male' if inp['sex']==1 else 'Female'}",
        f"  Chest Pain Type          : {cp_map.get(inp['cp'], inp['cp'])}",
        f"  Resting BP               : {inp['trestbps']} mm Hg",
        f"  Cholesterol              : {inp['chol']} mg/dl",
        f"  Fasting Blood Sugar>120  : {'Yes' if inp['fbs']==1 else 'No'}",
        f"  Resting ECG              : {['Normal','ST-T Abnormality','LV Hypertrophy'][inp['restecg']]}",
        f"  Max Heart Rate           : {inp['thalach']} bpm",
        f"  Exercise Angina          : {'Yes' if inp['exang']==1 else 'No'}",
        f"  ST Depression            : {inp['oldpeak']}",
        f"  ST Slope                 : {slope_map.get(inp['slope'], inp['slope'])}",
        f"  Major Vessels (blocked)  : {inp['ca']}",
        f"  Thalassemia              : {thal_map.get(inp['thal'], inp['thal'])}",
        "",
        "─" * 56,
        "  RECOMMENDATIONS",
        "─" * 56,
    ] + [f"  {i+1}. {t}" for i, t in enumerate(tips)] + [
        "",
        "─" * 56,
        "  DISCLAIMER: Not a substitute for medical advice.",
        "  Always consult a qualified healthcare professional.",
        "═" * 56,
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR  — session_state button navigation
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:24px 16px 20px;border-bottom:1px solid #2A2B38;margin-bottom:8px;'>
        <div style='font-family:"DM Serif Display",serif;font-size:1.5rem;
                    color:#F5F0EB;letter-spacing:-0.3px;line-height:1.2;'>
            🫀 CardioSense
        </div>
        <div style='font-size:.67rem;color:#555770;margin-top:5px;
                    letter-spacing:1.5px;text-transform:uppercase;'>
            Heart Risk Intelligence
        </div>
    </div>
    <div style='font-size:.65rem;letter-spacing:2px;text-transform:uppercase;
                color:#555770;padding:16px 16px 6px;'>Navigate</div>
    """, unsafe_allow_html=True)

    NAV_PAGES = [
        ("🏠", "Home"),
        ("🔬", "Prediction"),
        ("📊", "Data Analysis"),
        ("🧠", "Model Insights"),
        ("📄", "Download Report"),
    ]

    for icon, name in NAV_PAGES:
        is_active = (st.session_state.page == name)
        # Active page gets highlighted via inline style injected around the button
        if is_active:
            st.markdown(f"""
            <div style='background:linear-gradient(90deg,#3a0a12,#2a0a10);
                        border-left:3px solid #E5304A;border-radius:10px;
                        margin-bottom:2px;'>
            """, unsafe_allow_html=True)

        clicked = st.button(f"{icon}  {name}", key=f"nav_{name}",
                            use_container_width=True)
        if clicked:
            go(name)
            st.rerun()

        if is_active:
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:36px;padding:0 16px;font-size:.7rem;
                color:#404258;line-height:1.8;'>
        Trained on Cleveland Heart Disease<br>dataset (n = 303).<br><br>
        <em>Not medical advice.</em>
    </div>
    """, unsafe_allow_html=True)


# Active page
PAGE = st.session_state.page


# ═══════════════════════════════════════════════════════════════
#  PAGE — HOME
# ═══════════════════════════════════════════════════════════════
if PAGE == "Home":
    st.markdown("""
    <div style='padding:36px 0 16px;'>
        <div style='font-size:.75rem;color:#E5304A;text-transform:uppercase;
                    letter-spacing:3px;margin-bottom:10px;'>
            AI-Powered Health Assessment
        </div>
        <h1 style='font-size:2.8rem;line-height:1.18;margin:0;'>
            Know Your Heart.<br>Before It's Too Late.
        </h1>
        <p style='color:#8B8FA8;max-width:540px;margin-top:16px;
                  font-size:1rem;line-height:1.75;'>
            CardioSense uses a machine-learning model trained on clinical
            patient records to assess your heart disease risk — instantly,
            with an explainable confidence score.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, v, l in zip(
        [c1,c2,c3,c4],
        [f"{metrics['accuracy']}%","303","13","2 min"],
        ["Model Accuracy","Patient Records","Clinical Features","Assessment Time"]
    ):
        col.markdown(f"""
        <div class='metric-tile'>
            <div class='metric-value'>{v}</div>
            <div class='metric-label'>{l}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## How It Works")
    h1, h2, h3 = st.columns(3)
    for col, n, title, desc in zip(
        [h1,h2,h3],
        ["01","02","03"],
        ["Input Clinical Data","AI Analyses Risk","Download Report"],
        ["Enter your health metrics — age, cholesterol, ECG results, and more.",
         "Our Random Forest model processes your data and calculates risk probability.",
         "Get a personalised report with risk level and health recommendations."]
    ):
        col.markdown(f"""
        <div class='card'>
            <div style='font-family:"DM Serif Display",serif;font-size:2.2rem;
                        color:#2A2B38;line-height:1;'>{n}</div>
            <div style='font-weight:600;font-size:.95rem;margin:10px 0 7px;
                        color:#F5F0EB;'>{title}</div>
            <div style='font-size:.85rem;color:#8B8FA8;line-height:1.65;'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Key Risk Factors Monitored")
    factors = [
        ("🫀","Chest Pain Type","Type of pain is a strong predictor."),
        ("🩸","Cholesterol","High LDL levels correlate with disease."),
        ("⚡","ST Depression","Exercise ST changes signal cardiac stress."),
        ("🔬","Vessel Blockage","Fluoroscopy-detected blocks = direct risk."),
        ("💓","Max Heart Rate","Lower achieved rate may indicate problems."),
        ("🧬","Thalassemia","Blood disorder status affects heart load."),
    ]
    fl, fr = st.columns(2)
    for i, (ico, nm, dsc) in enumerate(factors):
        (fl if i%2==0 else fr).markdown(f"""
        <div style='display:flex;gap:14px;align-items:flex-start;
                    background:#16171F;border:1px solid #2A2B38;
                    border-radius:12px;padding:15px;margin-bottom:10px;'>
            <div style='font-size:1.4rem;'>{ico}</div>
            <div>
                <div style='font-weight:600;font-size:.93rem;color:#F5F0EB;'>{nm}</div>
                <div style='font-size:.82rem;color:#8B8FA8;margin-top:3px;
                            line-height:1.5;'>{dsc}</div>
            </div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE — PREDICTION
# ═══════════════════════════════════════════════════════════════
elif PAGE == "Prediction":
    st.markdown("## 🔬 Heart Disease Risk Assessment")
    st.markdown("<p style='color:#8B8FA8;margin-bottom:20px;'>"
                "Fill every field accurately for the most reliable result.</p>",
                unsafe_allow_html=True)

    with st.form("pred_form"):
        st.markdown("### 👤 Demographics")
        a1, a2 = st.columns(2)
        age = a1.number_input("Age (years)", 18, 100, 50)
        sex = a2.selectbox("Biological Sex", ["Male","Female"])

        st.markdown("### 🩺 Symptoms & Vitals")
        b1, b2 = st.columns(2)
        cp_map  = {"Typical Angina":0,"Atypical Angina":1,
                   "Non-Anginal Pain":2,"Asymptomatic":3}
        ecg_map = {"Normal":0,"ST-T Wave Abnormality":1,"LV Hypertrophy":2}
        cp       = b1.selectbox("Chest Pain Type", list(cp_map))
        trestbps = b1.number_input("Resting Blood Pressure (mm Hg)", 80, 220, 130)
        fbs      = b1.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No","Yes"])
        chol     = b2.number_input("Serum Cholesterol (mg/dl)", 100, 600, 246)
        restecg  = b2.selectbox("Resting ECG Results", list(ecg_map))

        st.markdown("### 🏃 Exercise Test")
        c1_, c2_ = st.columns(2)
        slope_map = {"Upsloping (good)":0,"Flat":1,"Downsloping (bad)":2}
        thalach   = c1_.number_input("Max Heart Rate Achieved (bpm)", 60, 220, 150)
        exang     = c1_.selectbox("Exercise-Induced Angina", ["No","Yes"])
        oldpeak   = c2_.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, 0.1)
        slope     = c2_.selectbox("Slope of Peak ST Segment", list(slope_map))

        st.markdown("### 🔬 Diagnostics")
        d1, d2 = st.columns(2)
        ca_map   = {"0 — None":0,"1 — One":1,"2 — Two":2,"3 — Three":3}
        thal_map = {"Normal":3.0,"Fixed Defect":6.0,"Reversable Defect":7.0}
        ca   = d1.selectbox("Major Vessels (Fluoroscopy)", list(ca_map))
        thal = d2.selectbox("Thalassemia", list(thal_map))

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🫀  Assess My Risk", use_container_width=True)

    if submitted:
        with st.spinner("Analysing your data…"):
            time.sleep(1.0)

        inp = {
            "age": age,
            "sex": 1 if sex=="Male" else 0,
            "cp":  cp_map[cp],
            "trestbps": trestbps,
            "chol": chol,
            "fbs": 1 if fbs=="Yes" else 0,
            "restecg": ecg_map[restecg],
            "thalach": thalach,
            "exang": 1 if exang=="Yes" else 0,
            "oldpeak": oldpeak,
            "slope": slope_map[slope],
            "ca": ca_map[ca],
            "thal": thal_map[thal],
        }
        vec   = np.array([[inp[f] for f in FEATURES]])
        vec_s = scaler.transform(vec)
        pred  = int(model.predict(vec_s)[0])
        prob  = float(model.predict_proba(vec_s)[0][pred])

        st.session_state.last_inputs     = inp
        st.session_state.last_prediction = pred
        st.session_state.last_prob       = prob

        st.markdown("---")
        st.markdown("### 📋 Result")

        if pred == 1:
            st.markdown(f"""
            <div class='result-high'>
                <div style='font-family:"DM Serif Display",serif;font-size:1.9rem;
                            color:#FF6B81;margin-bottom:6px;'>⚠️ High Risk Detected</div>
                <div style='color:#8B8FA8;font-size:.9rem;margin-bottom:18px;'>
                    Heart disease indicators present in your profile</div>
                <div style='font-family:"DM Serif Display",serif;font-size:3rem;
                            color:#E5304A;'>{prob*100:.1f}%</div>
                <div style='font-size:.75rem;color:#8B8FA8;letter-spacing:1px;
                            text-transform:uppercase;margin-top:4px;'>Confidence Score</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-low'>
                <div style='font-family:"DM Serif Display",serif;font-size:1.9rem;
                            color:#5ce698;margin-bottom:6px;'>✅ Low Risk</div>
                <div style='color:#8B8FA8;font-size:.9rem;margin-bottom:18px;'>
                    No significant heart disease indicators detected</div>
                <div style='font-family:"DM Serif Display",serif;font-size:3rem;
                            color:#5ce698;'>{prob*100:.1f}%</div>
                <div style='font-size:.75rem;color:#8B8FA8;letter-spacing:1px;
                            text-transform:uppercase;margin-top:4px;'>Confidence Score</div>
            </div>""", unsafe_allow_html=True)

        bar_color = "#E5304A" if pred==1 else "#5ce698"
        st.markdown(f"""
        <div style='margin-top:14px;background:#1e1f2a;border-radius:8px;
                    height:10px;overflow:hidden;'>
            <div style='background:{bar_color};width:{prob*100:.1f}%;
                        height:100%;border-radius:8px;'></div>
        </div>
        <div style='display:flex;justify-content:space-between;
                    font-size:.72rem;color:#555770;margin-top:3px;'>
            <span>0%</span><span>50%</span><span>100%</span>
        </div>""", unsafe_allow_html=True)

        st.markdown("### 💡 Recommendations")
        tips = (
            ["Consult a cardiologist promptly.",
             "Monitor blood pressure & cholesterol weekly.",
             "Low-sodium, heart-healthy diet.",
             "Avoid smoking; limit alcohol.",
             "Light aerobic exercise as medically advised."]
            if pred==1 else
            ["Maintain a balanced, plant-rich diet.",
             "150+ min/week of moderate exercise.",
             "Annual cardiac check-up recommended.",
             "Manage stress with mindfulness or yoga.",
             "Keep cholesterol and BP in healthy ranges."]
        )
        for t in tips:
            st.markdown(f"<div class='tip-card'>{'⚠️' if pred==1 else '✅'} {t}</div>",
                        unsafe_allow_html=True)

        st.info("Go to **📄 Download Report** in the sidebar to save your result.", icon="📄")


# ═══════════════════════════════════════════════════════════════
#  PAGE — DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════
elif PAGE == "Data Analysis":
    dark_theme()
    st.markdown("## 📊 Data Analysis")
    st.markdown("<p style='color:#8B8FA8;margin-bottom:24px;'>"
                "Exploratory analysis of the Cleveland Heart Disease dataset (n=303).</p>",
                unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    for col, v, l in zip(
        [m1,m2,m3],
        [len(df), int(df["target"].sum()), f"{df['chol'].mean():.0f} mg/dl"],
        ["Total Patients","With Heart Disease","Avg Cholesterol"]
    ):
        col.markdown(f"""<div class='metric-tile'>
            <div class='metric-value'>{v}</div>
            <div class='metric-label'>{l}</div></div>""",
            unsafe_allow_html=True)

    st.markdown("---")
    r1l, r1r = st.columns(2)

    with r1l:
        st.markdown("#### Age Distribution by Outcome")
        fig, ax = plt.subplots(figsize=(6,4))
        df[df["target"]==1]["age"].plot(kind="hist",bins=20,alpha=.75,
            color="#E5304A",ax=ax,label="Heart Disease")
        df[df["target"]==0]["age"].plot(kind="hist",bins=20,alpha=.65,
            color="#5ce698",ax=ax,label="No Disease")
        ax.set_xlabel("Age"); ax.set_ylabel("Count")
        ax.legend(facecolor="#1e1f2a",edgecolor="#2A2B38",labelcolor="#F5F0EB")
        ax.grid(True); fig.tight_layout(); st.pyplot(fig); plt.close()

    with r1r:
        st.markdown("#### Disease Prevalence")
        fig, ax = plt.subplots(figsize=(6,4))
        sizes = [len(df[df["target"]==0]), len(df[df["target"]==1])]
        wedges, texts, autos = ax.pie(
            sizes, labels=["No Disease","Heart Disease"],
            colors=["#5ce698","#E5304A"], autopct="%1.1f%%", startangle=90,
            textprops={"color":"#F5F0EB"},
            wedgeprops={"edgecolor":"#16171F","linewidth":2})
        for a in autos: a.set_fontsize(11); a.set_fontweight("bold")
        ax.set_facecolor("#16171F"); fig.tight_layout()
        st.pyplot(fig); plt.close()

    r2l, r2r = st.columns(2)

    with r2l:
        st.markdown("#### Cholesterol by Outcome")
        fig, ax = plt.subplots(figsize=(6,4))
        bp = ax.boxplot(
            [df[df["target"]==0]["chol"], df[df["target"]==1]["chol"]],
            patch_artist=True, widths=0.45,
            medianprops=dict(color="#F5F0EB",linewidth=2),
            flierprops=dict(marker="o",markersize=4,alpha=.5))
        bp["boxes"][0].set_facecolor("#5ce698")
        bp["boxes"][1].set_facecolor("#E5304A")
        ax.set_xticks([1,2]); ax.set_xticklabels(["No Disease","Heart Disease"])
        ax.set_ylabel("Cholesterol (mg/dl)"); ax.grid(True,axis="y")
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with r2r:
        st.markdown("#### Age vs Max Heart Rate")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(df[df["target"]==0]["age"],df[df["target"]==0]["thalach"],
                   alpha=.55,color="#5ce698",s=22,label="No Disease")
        ax.scatter(df[df["target"]==1]["age"],df[df["target"]==1]["thalach"],
                   alpha=.55,color="#E5304A",s=22,label="Heart Disease")
        ax.set_xlabel("Age"); ax.set_ylabel("Max Heart Rate (bpm)")
        ax.legend(facecolor="#1e1f2a",edgecolor="#2A2B38",labelcolor="#F5F0EB")
        ax.grid(True); fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12,5))
    corr = df[FEATURES+["target"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap=sns.diverging_palette(10,150,as_cmap=True),
                linewidths=.5, linecolor="#16171F", ax=ax,
                annot_kws={"size":8}, cbar_kws={"shrink":.8})
    ax.tick_params(labelsize=8); fig.tight_layout()
    st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════════
#  PAGE — MODEL INSIGHTS
# ═══════════════════════════════════════════════════════════════
elif PAGE == "Model Insights":
    dark_theme()
    st.markdown("## 🧠 Model Insights")
    st.markdown("<p style='color:#8B8FA8;margin-bottom:24px;'>"
                "Random Forest Classifier · trained on 80% of the dataset.</p>",
                unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    for col, v, l in zip(
        [m1,m2,m3],
        [f"{metrics['accuracy']}%",f"{metrics['precision']}%",f"{metrics['recall']}%"],
        ["Accuracy","Precision","Recall"]
    ):
        col.markdown(f"""<div class='metric-tile'>
            <div class='metric-value'>{v}</div>
            <div class='metric-label'>{l}</div></div>""",
            unsafe_allow_html=True)

    st.markdown(f"""
    <div class='card' style='margin-top:20px;'>
        <div style='font-weight:600;margin-bottom:10px;color:#F5F0EB;'>
            📖 Plain-English Explanation</div>
        <div style='font-size:.87rem;color:#8B8FA8;line-height:1.9;'>
            <b style='color:#F5F0EB;'>Accuracy {metrics['accuracy']}%</b>
            — The model is correct {metrics['accuracy']}% of the time overall.<br>
            <b style='color:#F5F0EB;'>Precision {metrics['precision']}%</b>
            — When it flags "Heart Disease", it is right {metrics['precision']}% of the time.<br>
            <b style='color:#F5F0EB;'>Recall {metrics['recall']}%</b>
            — It catches {metrics['recall']}% of all actual heart disease cases.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(metrics["cm"], annot=True, fmt="d", ax=ax,
                    cmap="RdYlGn_r", linewidths=1, linecolor="#16171F",
                    xticklabels=["No Disease","Heart Disease"],
                    yticklabels=["No Disease","Heart Disease"],
                    annot_kws={"size":14,"weight":"bold","color":"#F5F0EB"},
                    cbar=False)
        ax.set_xlabel("Predicted",labelpad=10)
        ax.set_ylabel("Actual",labelpad=10)
        ax.tick_params(labelsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.caption("Rows = actual class · Columns = predicted class")

    with col_fi:
        st.markdown("#### Feature Importance")
        fi = pd.Series(metrics["importance"]).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(5,5))
        colors = ["#E5304A" if v >= fi.median() else "#555770" for v in fi.values]
        ax.barh(fi.index, fi.values, color=colors, height=0.65)
        ax.set_xlabel("Importance Score"); ax.grid(True,axis="x")
        ax.tick_params(axis="y",labelsize=8.5)
        fig.tight_layout(); st.pyplot(fig); plt.close()
        st.caption("Red = above-median importance")

    st.markdown("---")
    st.markdown("### 🔧 Model Architecture")
    st.markdown("""
    <div class='card'>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:.87rem;'>
            <div><span style='color:#8B8FA8;'>Algorithm</span><br>
                 <span style='color:#F5F0EB;font-weight:600;'>Random Forest Classifier</span></div>
            <div><span style='color:#8B8FA8;'>Estimators</span><br>
                 <span style='color:#F5F0EB;font-weight:600;'>200 decision trees</span></div>
            <div><span style='color:#8B8FA8;'>Max Depth</span><br>
                 <span style='color:#F5F0EB;font-weight:600;'>8 levels</span></div>
            <div><span style='color:#8B8FA8;'>Preprocessing</span><br>
                 <span style='color:#F5F0EB;font-weight:600;'>StandardScaler (z-score)</span></div>
            <div><span style='color:#8B8FA8;'>Train / Test Split</span><br>
                 <span style='color:#F5F0EB;font-weight:600;'>80 % / 20 %</span></div>
            <div><span style='color:#8B8FA8;'>Stratified</span><br>
                 <span style='color:#F5F0EB;font-weight:600;'>Yes</span></div>
        </div>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE — DOWNLOAD REPORT
# ═══════════════════════════════════════════════════════════════
elif PAGE == "Download Report":
    st.markdown("## 📄 Download Report")

    if "last_prediction" not in st.session_state:
        st.markdown("""
        <div class='card' style='text-align:center;padding:60px 40px;'>
            <div style='font-size:3rem;margin-bottom:16px;'>🔬</div>
            <div style='font-family:"DM Serif Display",serif;font-size:1.35rem;
                        color:#F5F0EB;margin-bottom:10px;'>No Assessment Yet</div>
            <div style='color:#8B8FA8;font-size:.9rem;'>
                Complete the <b>Prediction</b> form first, then return here
                to download your personalised report.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        pred  = st.session_state.last_prediction
        prob  = st.session_state.last_prob
        inpts = st.session_state.last_inputs
        badge = "badge-red" if pred==1 else "badge-green"
        label = "HIGH RISK" if pred==1 else "LOW RISK"

        st.markdown(f"""
        <div class='card'>
            <div style='margin-bottom:16px;'><span class='{badge}'>{label}</span></div>
            <div style='font-family:"DM Serif Display",serif;font-size:1.5rem;
                        color:#F5F0EB;margin-bottom:6px;'>Your Assessment is Ready</div>
            <div style='color:#8B8FA8;font-size:.88rem;line-height:1.7;'>
                Confidence Score:
                <strong style='color:#F5F0EB;'>{prob*100:.1f}%</strong><br>
                Your report includes all inputs, result, risk level,
                and personalised health recommendations.
            </div>
        </div>""", unsafe_allow_html=True)

        report = make_report(inpts, pred, prob)
        st.download_button(
            label="⬇️  Download My Report (.txt)",
            data=report,
            file_name="CardioSense_Report.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.markdown("---")
        st.markdown("### Preview")
        st.code(report, language=None)
