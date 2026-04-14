import streamlit as st
import pandas as pd
import numpy as np

# Title
st.title('Heart Disease Prediction App')

# Sidebar
st.sidebar.header('User Input Parameters')

# Home Page
if st.sidebar.radio('Navigation', ['Home', 'Prediction', 'Model Insights', 'Data Analysis']) == 'Home':
    st.write('Welcome to the Heart Disease Prediction App!')
    st.write('This application allows you to predict heart disease based on user input.')

# Prediction Page
elif st.sidebar.radio('Navigation', ['Home', 'Prediction', 'Model Insights', 'Data Analysis']) == 'Prediction':
    st.subheader('Prediction')
    # User Input Features
    age = st.number_input('Age', min_value=0)
    sex = st.selectbox('Sex', options=[0, 1])  # 0 = female, 1 = male
    cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure', min_value=0)
    chol = st.number_input('Serum Cholestoral', min_value=0)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
    restecg = st.selectbox('Resting ECG Results', options=[0, 1, 2])
    thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0)
    exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
    oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2])
    ca = st.selectbox('Number of Major Vessels (0-3)', options=[0, 1, 2, 3])
    thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3])
    
    # Prediction Logic
    if st.button('Predict'):
        # Placeholder for model prediction (replace with actual model call)
        prediction = np.random.choice([0, 1])
        st.success('Prediction: Heart Disease' if prediction == 1 else 'No Heart Disease')

# Model Insights Page
elif st.sidebar.radio('Navigation', ['Home', 'Prediction', 'Model Insights', 'Data Analysis']) == 'Model Insights':
    st.subheader('Model Insights')
    st.write('Insights and visualizations about the model performance, feature importance, etc.')

# Data Analysis Page
elif st.sidebar.radio('Navigation', ['Home', 'Prediction', 'Model Insights', 'Data Analysis']) == 'Data Analysis':
    st.subheader('Data Analysis')
    st.write('Exploratory Data Analysis (EDA) on heart disease dataset.')
    # Placeholder for data visualization code and analysis here