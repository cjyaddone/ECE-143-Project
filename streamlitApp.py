import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# ---------------------------------------------------------
# Load the saved full pipeline model (preprocessing + model)
# ---------------------------------------------------------

MODEL_PATH = "saved_models/gradient_boosting.joblib"
model = joblib.load(MODEL_PATH)

# SHAP-safe wrapper to avoid feature_names_in_ errors
def make_shap_predictor(feature_names):
    def _predict(X):
        X_df = pd.DataFrame(X, columns=feature_names)
        return model.predict(X_df)
    return _predict


st.set_page_config(page_title="Sleep Quality Predictor", page_icon="😴", layout="wide")

st.title("😴 Sleep Quality Prediction App")
st.write("Enter your lifestyle and health information to estimate your sleep quality score (1–10).")
st.write("---")


# ---------------------------------------------------------
# Helper values (for optional inputs)
# ---------------------------------------------------------
DEFAULT_SLEEP_DURATION = 7.0
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
    physical_activity = st.number_input(
        "Physical Activity (minutes/day)", min_value=0, max_value=300, value=60
    )
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

    # Make prediction
    prediction = model.predict(input_df)[0]
    st.success(f"### ⭐ Your Predicted Sleep Quality Score: **{prediction:.2f} / 10**")

    # -----------------------------
    # SHAP EXPLAINER (kernel)
    # -----------------------------
    predict_fn = make_shap_predictor(input_df.columns.tolist())
    background = input_df.copy()

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(input_df)
    shap_row = shap_values[0]

    abs_shap = np.abs(shap_row)
    feature_names = input_df.columns.tolist()
    ranking_idx = np.argsort(abs_shap)[::-1]

    # ----------------------------------------------------------
    # MODIFIABLE FEATURES — These are the ones the user can fix
    # ----------------------------------------------------------
    modifiable_features = [
        "Stress Level",
        "Physical Activity Level",
        "Daily Steps",
        "Heart Rate",
    ]

    # FIRST FEATURE = Sleep Duration
    first_feature = "Sleep Duration"

    # SECOND FEATURE = top SHAP modifiable feature
    second_feature = None
    for idx in ranking_idx:
        fname = feature_names[idx]
        if fname in modifiable_features:
            second_feature = fname
            break

    if second_feature is None:
        second_feature = "Stress Level"  # safe fallback

    # -----------------------------
    # Natural language interpretation
    # -----------------------------
    if prediction >= 8:
        st.info("Great! Your predicted sleep quality is very high. Your lifestyle factors seem supportive of good sleep.")
    elif prediction >= 6:
        st.info("Your sleep quality seems solid, but improvements to stress or sleep habits may help.")
    elif prediction >= 4:
        st.warning("Your predicted sleep quality is moderate. Consider adjusting sleep duration or reducing stress.")
    else:
        st.error("Your predicted sleep quality is low. Stress reduction and healthier routines may help significantly.")

    st.write("---")

    # ===========================================================
    # Personalized Improvement Suggestions
    # ===========================================================

    st.subheader("💡 How You Can Improve Your Sleep Score")

    suggestions = []

    # Sleep Duration suggestion
    if sleep_duration < 7:
        suggestions.append(
            f"- Increase your sleep duration closer to **7–8 hours**. You currently sleep **{sleep_duration} hours**."
        )
    elif sleep_duration > 9:
        suggestions.append(
            f"- Your sleep duration (**{sleep_duration} hours**) is higher than average. "
            f"Most adults sleep best around **7–8 hours**."
        )
    else:
        suggestions.append(f"- Your sleep duration (**{sleep_duration} hours**) is healthy.")

    # Dynamic second feature suggestion
    value = input_df[second_feature].iloc[0]

    if second_feature == "Stress Level":
        if value >= 7:
            suggestions.append(f"- Your **Stress Level = {value}/10** is high. Reducing stress could greatly improve sleep quality.")
        elif value >= 5:
            suggestions.append(f"- Your **Stress Level = {value}/10** is moderate. Relaxation routines may help.")
        else:
            suggestions.append(f"- Your stress level (**{value}/10**) is low, which supports good sleep.")

    elif second_feature == "Daily Steps":
        if value < 6000:
            suggestions.append(f"- Your steps (**{value}**) are below ideal. Increasing to **8k–10k steps/day** may improve sleep.")
        else:
            suggestions.append(f"- Your step count (**{value}**) is healthy.")

    elif second_feature == "Physical Activity Level":
        if value < 30:
            suggestions.append(f"- Your physical activity (**{value} min/day**) is low. Increasing it may improve sleep.")
        else:
            suggestions.append(f"- Your physical activity (**{value} min/day**) is healthy.")

    elif second_feature == "Heart Rate":
        if value > 85:
            suggestions.append(f"- Your resting heart rate (**{value} bpm**) is elevated. Improving cardio fitness may help.")
        else:
            suggestions.append(f"- Your heart rate (**{value} bpm**) is typical and healthy.")

    # Display suggestions
    for s in suggestions:
        st.write(s)

    st.write("---")
    st.subheader("🔍 Your Input Summary")
    st.dataframe(input_df)
