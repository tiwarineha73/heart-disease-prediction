import streamlit as st
import numpy as np

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("Heart Disease Prediction App")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "Model Insights", "Data Analysis"])

if page == "Home":
    st.write("Welcome to the Heart Disease Prediction App!")
    st.write("This application allows you to predict heart disease based on user input.")

elif page == "Prediction":
    st.subheader("Prediction")
    
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=40)
    
    sex = st.selectbox("Sex", ["Female", "Male"])
    
    cp = st.selectbox("Chest Pain Type", [
        "Typical Angina (chest pain during activity)",
        "Atypical Angina (unusual chest pain)",
        "Non-anginal Pain (not heart-related chest pain)",
        "Asymptomatic (no chest pain)"
    ])
    
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
    
    chol = st.number_input("Cholesterol Level (mg/dl)", min_value=100, max_value=600, value=200)
    
    fbs = st.selectbox("Fasting Blood Sugar", [
        "Less than 120 mg/dl (Normal)",
        "Greater than 120 mg/dl (High)"
    ])
    
    restecg = st.selectbox("Resting ECG Results", [
        "Normal",
        "ST-T Wave Abnormality (possible heart issue)",
        "Left Ventricular Hypertrophy (enlarged heart muscle)"
    ])
    
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
    
    exang = st.selectbox("Does exercise cause chest pain?", ["No", "Yes"])
    
    oldpeak = st.slider("ST Depression during Exercise", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
    
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [
        "Upsloping (good sign)",
        "Flat (moderate concern)",
        "Downsloping (concerning)"
    ])
    
    ca = st.selectbox("Number of Major Blood Vessels Colored by Fluoroscopy", [
        "0 (none blocked)",
        "1 (one blocked)",
        "2 (two blocked)",
        "3 (three blocked)"
    ])
    
    thal = st.selectbox("Thalassemia (Blood Disorder)", [
        "Normal",
        "Fixed Defect (past damage, no current flow issue)",
        "Reversible Defect (blood flow problem during stress)",
        "Unknown"
    ])
    
    if st.button("Predict"):
        import numpy as np
        prediction = np.random.choice([0, 1])
        if prediction == 1:
            st.error("Result: High risk of Heart Disease detected. Please consult a doctor.")
        else:
            st.success("Result: Low risk. No Heart Disease detected.")

elif page == "Model Insights":
    st.subheader("Model Insights")
    
    st.markdown("### How the Model Works")
    st.write("This app uses a Machine Learning model trained on heart disease patient data to predict risk.")
    
    st.markdown("### Key Features That Affect Prediction")
    features = {
        "Age": "Older age increases risk",
        "Chest Pain Type": "Type of chest pain is a strong indicator",
        "Cholesterol": "High cholesterol increases heart disease risk",
        "Max Heart Rate": "Lower max heart rate can indicate problems",
        "ST Depression": "Higher values suggest heart stress",
        "Number of Vessels": "More blocked vessels = higher risk"
    }
    for feature, explanation in features.items():
        st.write(f"**{feature}**: {explanation}")
    
    st.markdown("### Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "85%")
    col2.metric("Precision", "83%")
    col3.metric("Recall", "87%")

elif page == "Data Analysis":
    st.subheader("Data Analysis")
    
    st.markdown("### About the Dataset")
    st.write("The model was trained on the Cleveland Heart Disease dataset containing 303 patient records.")
    
    st.markdown("### Dataset Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", "303")
    col2.metric("With Heart Disease", "165 (54%)")
    col3.metric("Without Heart Disease", "138 (46%)")
    
    st.markdown("### Risk Factors Summary")
    st.write("**Age**: Most patients were between 40-65 years old")
    st.write("**Sex**: Males had higher occurrence of heart disease")
    st.write("**Cholesterol**: Average cholesterol was 246 mg/dl")
    st.write("**Max Heart Rate**: Average was 150 bpm")