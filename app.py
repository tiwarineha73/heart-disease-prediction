import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model
model_filename = 'random_forest_model.pkl'
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('heart_failure.csv')
    return data

df = load_data()

st.title('Heart Disease Prediction App')

# Display dataset insights
if st.checkbox('Show Dataset Insights'):
    st.subheader('Dataset Preview')
    st.write(df.head())
    st.subheader('Statistical Summary')
    st.write(df.describe())
    st.subheader('Feature Importance')
    feature_importance = pd.Series(model.feature_importances_, index=df.columns[:-1]).sort_values(ascending=False)
    st.bar_chart(feature_importance)

# Define user input fields
age = st.number_input('Age', min_value=1, max_value=120)
anaemia = st.selectbox('Anaemia', [0, 1])
creatinine_phosphokinase = st.number_input('Creatinine Phosphokinase', min_value=0)
diabetes = st.selectbox('Diabetes', [0, 1])
ejection_fraction = st.number_input('Ejection Fraction', min_value=0, max_value=100)
high_blood_pressure = st.selectbox('High Blood Pressure', [0, 1])
platelets = st.number_input('Platelets', min_value=0)
serum_creatinine = st.number_input('Serum Creatinine', min_value=0.0)
serum_sodium = st.number_input('Serum Sodium', min_value=0.0)
sex = st.selectbox('Sex', [0, 1])
smoking = st.selectbox('Smoking', [0, 1])
time = st.number_input('Time', min_value=0)

# Create input data DataFrame
input_data = pd.DataFrame({
    'age': [age],
    'anaemia': [anaemia],
    'creatinine_phosphokinase': [creatinine_phosphokinase],
    'diabetes': [diabetes],
    'ejection_fraction': [ejection_fraction],
    'high_blood_pressure': [high_blood_pressure],
    'platelets': [platelets],
    'serum_creatinine': [serum_creatinine],
    'serum_sodium': [serum_sodium],
    'sex': [sex],
    'smoking': [smoking],
    'time': [time]
})

# Predict mortality risk
if st.button('Predict Mortality Risk'):
    try:
        prediction = model.predict_proba(input_data)[:, 1][0]
        st.success(f'Mortality Risk Probability: {prediction:.2f}')
    except Exception as e:
        st.error(f'Error in prediction: {str(e)}')

# Additional visualizations
if st.checkbox('Show Additional Visualizations'):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    st.pyplot()