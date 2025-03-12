import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from scipy.stats.mstats import winsorize

# Define the custom function
def log1p_transform(X):
    return np.log1p(X)

# Define the custom function
def winsorize_transform(X):
    return np.array([winsorize(col, limits=[0.05, 0.05]) for col in X.T]).T


# Load the pre-trained pipeline
with open('chd.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Configure page settings
st.set_page_config(
    page_title="CHD Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stNumberInput, .stSelectbox {background-color: white;}
    .prediction-box {padding: 20px; border-radius: 10px; margin-top: 20px;}
    </style>
    """, unsafe_allow_html=True)

# App header
st.title("10-Year Coronary Heart Disease Risk Prediction")
st.markdown("""
This app predicts the 10-year risk of developing coronary heart disease (CHD) using the Framingham Heart Study dataset.
""")

# Create input columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographic Information")
    age = st.number_input("Age (years)", min_value=20, max_value=100, value=50)
    male = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    education = st.selectbox(
        "Education Level",
        options=[1, 2, 3, 4],
        help="1: Some High School, 2: High School/GED, 3: Some College, 4: College"
    )

with col2:
    st.subheader("Health Metrics")
    sysBP = st.number_input("Systolic BP (mmHg)", min_value=90, max_value=250, value=120)
    diaBP = st.number_input("Diastolic BP (mmHg)", min_value=60, max_value=150, value=80)
    BMI = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
    totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=200)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=100)

with col3:
    st.subheader("Medical History & Habits")
    currentSmoker = st.selectbox("Current Smoker", options=[0, 1], 
                               format_func=lambda x: "No" if x == 0 else "Yes")
    cigsPerDay = st.number_input("Cigarettes per Day", min_value=0, max_value=100, value=0,
                                help="Enter 0 if not smoking")
    BPMeds = st.selectbox("Blood Pressure Medication", options=[0, 1],
                        format_func=lambda x: "No" if x == 0 else "Yes")
    prevalentStroke = st.selectbox("History of Stroke", options=[0, 1],
                                 format_func=lambda x: "No" if x == 0 else "Yes")
    prevalentHyp = st.selectbox("Hypertension", options=[0, 1],
                              format_func=lambda x: "No" if x == 0 else "Yes")
    diabetes = st.selectbox("Diabetes", options=[0, 1],
                          format_func=lambda x: "No" if x == 0 else "Yes")

# Create feature dictionary
input_data = {
    'male': male,
    'education': education,
    'currentSmoker': currentSmoker,
    'BPMeds': BPMeds,
    'prevalentStroke': prevalentStroke,
    'prevalentHyp': prevalentHyp,
    'diabetes': diabetes,
    'age': age,
    'cigsPerDay': cigsPerDay,
    'totChol': totChol,
    'sysBP': sysBP,
    'diaBP': diaBP,
    'BMI': BMI,
    'glucose': glucose
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction button
if st.button("Predict CHD Risk"):
    try:
        # Make prediction
        prediction = pipeline.predict(input_df)
        probability = pipeline.predict_proba(input_df)[0][1]
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Result")
        
        if prediction[0] == 1:
            result = f"**High Risk** (Probability: {probability:.1%})"
            st.markdown(f"<div class='prediction-box' style='background-color: #ffcccc'>{result}</div>", 
                      unsafe_allow_html=True)
            st.warning("This individual has a high risk of developing coronary heart disease within 10 years. \
                      Please consult with a healthcare professional for further evaluation.")
        else:
            result = f"**Low Risk** (Probability: {probability:.1%})"
            st.markdown(f"<div class='prediction-box' style='background-color: #ccffcc'>{result}</div>", 
                      unsafe_allow_html=True)
            st.success("This individual has a low risk of developing coronary heart disease within 10 years. \
                      Maintain a healthy lifestyle for continued well-being.")
        
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
**Disclaimer**: This prediction tool is based on statistical models and should not replace professional medical advice. 
Always consult with a qualified healthcare provider for health-related decisions.
""")