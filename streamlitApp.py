import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------------
# Load the saved full pipeline model (preprocessing + model)
# ---------------------------------------------------------

MODEL_PATH = "saved_models/gradient_boosting.joblib"
model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Sleep Quality Predictor", page_icon="😴", layout="wide")

st.title("😴 Sleep Quality Prediction App")
st.write("Enter your lifestyle and health information to estimate your sleep quality score (1–10).")
st.write("---")


# ---------------------------------------------------------
# Helper values (for optional inputs)
# ---------------------------------------------------------
DEFAULT_SLEEP_DURATION = 7.0   # median or trained model mean
st.sidebar.header("ℹ️ Optional Input Logic")
st.sidebar.write(f"If Sleep Duration is not provided, the app will assume **{DEFAULT_SLEEP_DURATION} hours**.")


# ---------------------------------------------------------
# User Input Form
# ---------------------------------------------------------

st.subheader("📋 Input Your Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=30)

    sleep_duration_input = st.text_input("Sleep Duration (hours, optional)", value="")
    stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=75)

with col2:
    physical_activity = st.number_input("Physical Activity (minutes/day)", min_value=0, max_value=300, value=60)
    daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=8000)

    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese"])

occupation = st.selectbox(
    "Occupation",
    [
        "Engineer", "Doctor", "Nurse", "Teacher", "Scientist", "Accountant",
        "Salesperson", "Sales Representative", "Software Engineer", "Lawyer",
        "Manager", "Other"
    ]
)

sleep_disorder = st.selectbox(
    "Sleep Disorder",
    ["None", "Sleep Apnea", "Insomnia"]
)

st.write("---")


# ---------------------------------------------------------
# Prepare Input DataFrame
# ---------------------------------------------------------

# Handle optional sleep duration
if sleep_duration_input.strip() == "":
    sleep_duration = DEFAULT_SLEEP_DURATION
    st.info(f"Sleep Duration not provided → using default value: **{DEFAULT_SLEEP_DURATION} hours**")
else:
    try:
        sleep_duration = float(sleep_duration_input)
        if sleep_duration <= 0 or sleep_duration > 24:
            st.warning("Sleep Duration must be between 0 and 24. Using default instead.")
            sleep_duration = DEFAULT_SLEEP_DURATION
    except:
        st.warning("Invalid Sleep Duration. Using default instead.")
        sleep_duration = DEFAULT_SLEEP_DURATION

# Build input row
input_dict = {
    "Age": age,
    "Sleep Duration": sleep_duration,
    "Physical Activity Level": physical_activity,
    "Stress Level": stress_level,
    "Heart Rate": heart_rate,
    "Daily Steps": daily_steps,
    "Gender": gender,
    "Occupation": occupation,
    "BMI Category": bmi_category,
    "Sleep Disorder": sleep_disorder
}

input_df = pd.DataFrame([input_dict])


# ---------------------------------------------------------
# Predict button
# ---------------------------------------------------------

if st.button("Predict Sleep Quality 😴"):
    prediction = model.predict(input_df)[0]

    st.success(f"### ⭐ Your Predicted Sleep Quality Score: **{prediction:.2f} / 10**")

    # Simple natural language interpretation
    if prediction >= 8:
        st.info("Great! Your predicted sleep quality is very high. Your lifestyle factors seem supportive of good sleep.")
    elif prediction >= 6:
        st.info("Your sleep quality seems solid, but improvements to stress or sleep habits may help.")
    elif prediction >= 4:
        st.warning("Your predicted sleep quality is moderate. Consider adjusting sleep duration or reducing stress.")
    else:
        st.error("Your predicted sleep quality is low. Stress reduction and healthier routines may help significantly.")

    st.write("---")
    st.subheader("🔍 Your Input Summary")
    st.dataframe(input_df)

