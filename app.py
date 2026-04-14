import streamlit as st
import pandas as pd
import numpy as np

# Title
st.title('Heart Disease Prediction App')

# Sidebar - ONE radio call stored in variable
st.sidebar.header('Navigation')
page = st.sidebar.radio('Go to', ['Home', 'Prediction', 'Model Insights', 'Data Analysis'])

# Home Page
if page == 'Home':
    st.write('Welcome to the Heart Disease Prediction App!')
    st.write('This application allows you to predict heart disease based on user input.')

# Prediction Page
elif page == 'Prediction':
    st.subheader('Prediction')
    age = st.number_input('Age', min_value=0)
    sex = st.selectbox('Sex', options=[0, 1])
    cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure', min_value=0)
    chol = st.number_input('Serum Cholestoral', min_value=0)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
    restecg = st.selectbox('Resting ECG Results', options=[0, 1, 2])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0)
    exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0)
    slope = st.selectbox('Slope of Peak Exercise ST Segment', options=[0, 1, 2])
    ca = st.selectbox('Number of Major Vessels (0-3)', options=[0, 1, 2, 3])
    thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3])

    if st.button('Predict'):
        prediction = np.random.choice([0, 1])
        if prediction == 1:
            st.error('Prediction: Heart Disease Detected')
        else:
            st.success('Prediction: No Heart Disease')

# Model Insights Page
elif page == 'Model Insights':
    st.subheader('Model Insights')
    st.write('Insights and visualizations about model performance, feature importance, etc.')

# Data Analysis Page
elif page == 'Data Analysis':
    st.subheader('Data Analysis')
    st.write('Exploratory Data Analysis (EDA) on heart