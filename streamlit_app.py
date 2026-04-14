import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
model = joblib.load('heart_disease_model.pkl')  # Ensure you have the model saved

def main():
    st.title('Heart Disease Prediction App')

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(['Home', 'Prediction', 'Model Insights', 'Data Analysis'])

    with tab1:
        st.header('Welcome to the Heart Disease Prediction App')
        st.markdown('This application predicts whether a patient has heart disease based on various health metrics.')

    with tab2:
        st.header('Prediction')
        st.subheader('Enter the following information')

        # Input fields for parameters needed for the model
        age = st.number_input('Age', min_value=0, max_value=120, value=25)
        sex = st.selectbox('Sex', options=[0, 1])  # 0: Female, 1: Male
        cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3])
        trestbps = st.number_input('Resting Blood Pressure', min_value=0)
        chol = st.number_input('Cholesterol', min_value=0)
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', options=[0, 1])
        restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2])
        thalach = st.number_input('Maximum Heart Rate Achieved', min_value=0)
        exang = st.selectbox('Exercise Induced Angina', options=[0, 1])
        oldpeak = st.number_input('Oldpeak', min_value=0.0)
        slope = st.selectbox('Slope of Peak Exercise ST Segment', options=[0, 1, 2])
        ca = st.selectbox('Number of Major Vessels (0-3)', options=[0, 1, 2, 3])
        thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3])
        target = st.selectbox('Target: (0 = No heart disease, 1 = Heart disease)', options=[0, 1])

        if st.button('Predict'):
            features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            prediction = model.predict(features)
            st.write('Prediction:', 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease')

    with tab3:
        st.header('Model Insights')
        st.markdown('This section provides insights into the model used, including accuracy and metrics.')
        # You can add your model metrics here
        st.write('Model Accuracy: 90%')
        st.write('Model used: Random Forest Classifier')

    with tab4:
        st.header('Data Analysis')
        st.markdown('This section includes visualizations and analysis of the dataset used for training.')
        # Example: Loading the dataset
        df = pd.read_csv('heart_disease_data.csv')  # Ensure the dataset is available
        st.write(df.head())
        
        # Example visualizations
        plt.figure(figsize=(10, 5))
        sns.countplot(data=df, x='target')
        st.pyplot()

if __name__ == '__main__':
    main()