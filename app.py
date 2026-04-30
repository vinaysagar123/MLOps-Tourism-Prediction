import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.title("Wellness Tourism Package Prediction")

# Load model from Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id="Sagar-143/wellness-tourism-model", filename="wellness_rf_model.joblib")
    return joblib.load(model_path)

model = load_model()

# User Inputs
st.header("Enter Customer Details")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=15)
monthly_income = st.number_input("Monthly Income", min_value=0, max_value=200000, value=50000)

if st.button("Predict Purchase"):
    # Save inputs to dataframe
    input_data = {
        'Age': [age],
        'CityTier': [city_tier],
        'DurationOfPitch': [duration_pitch],
        'MonthlyIncome': [monthly_income],
    }
    df_input = pd.DataFrame(input_data)
    
    st.success("Inputs captured and Model loaded successfully from Hugging Face!")
