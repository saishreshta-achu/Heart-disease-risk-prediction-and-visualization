import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------
# PAGE CONFIG
# -------------------

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

# -------------------
# CUSTOM THEME
# -------------------

st.markdown("""
<style>

.stApp{
    background-color:#0a0a0a;
    color:white;
}

section[data-testid="stSidebar"]{
    background-color:#1a0000;
}

h1,h2,h3{
    color:#ff4b4b !important;
}

div[data-testid="metric-container"]{
    background:#1c1c1c;
    border:1px solid #ff4b4b;
    border-radius:12px;
    padding:15px;
}

.stButton button{
    background-color:#ff4b4b;
    color:white;
    border:none;
    border-radius:8px;
}

</style>
""", unsafe_allow_html=True)

# -------------------
# LOAD MODEL
# -------------------

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")

# -------------------
# SESSION STATE INIT  [BUG FIX #5] — initialize history once at top level
# -------------------

if "history" not in st.session_state:
    st.session_state["history"] = []

# -------------------
# SIDEBAR
# -------------------

st.sidebar.title("❤️ Heart Risk AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Patient View",
        "Doctor View",
        "Analytics",
        "Bulk Prediction",
        "Prediction History",
        "Model Info",
        "About Project"
    ]
)

# ==========================
# DASHBOARD
# ==========================

if page == "Dashboard":

    df = pd.read_csv("heart.csv")

    st.markdown("""
    <h1 style='text-align:center;color:#ff4b4b;'>
    ❤️ Heart Disease Risk Prediction System
    </h1>

    <h4 style='text-align:center;color:white;'>
    AI Powered Cardiovascular Risk Assessment Dashboard
    </h4>
    """, unsafe_allow_html=True)

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "👥 Total Patients",
        len(df)
    )

    c2.metric(
        "📅 Average Age",
        round(df["Age"].mean(), 1)
    )

    c3.metric(
        "🫀 Avg Cholesterol",
        round(df["Cholesterol"].mean(), 1)
    )

    c4.metric(
        "⚖️ Avg BMI",
        round(df["BMI"].mean(), 1)
    )

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Age Distribution")
        st.bar_chart(df["Age"])

    with right:
        st.subheader("Cholesterol Distribution")
        st.bar_chart(df["Cholesterol"])

    st.markdown("---")

    st.subheader("Project Overview")

    st.info("""
This system predicts heart disease risk using machine learning
based on patient clinical parameters such as Age,
Blood Pressure, Cholesterol, BMI, Glucose,
Smoking Habits, Diabetes and Family History.
""")

# ==========================
# PATIENT VIEW
# ==========================

elif page == "Patient View":

    st.title("🧑 Patient Risk Assessment")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        age = st.number_input(
            "Age",
            18,
            100,
            50
        )

        glucose = st.number_input(
            "Glucose",
            50,
            300,
            100
        )

        bp = st.number_input(
            "Systolic BP",
            80,
            250,
            120
        )

    with col2:

        chol = st.number_input(
            "Cholesterol",
            100,
            500,
            200
        )

        bmi = st.number_input(
            "BMI",
            10.0,
            60.0,
            25.0
        )

        heart_rate = st.number_input(
            "Heart Rate",
            40,
            200,
            75
        )

        smoking = st.selectbox(
            "Smoking",
            ["No", "Yes"]
        )

        diabetes = st.selectbox(
            "Diabetes",
            ["No", "Yes"]
        )

        family_history = st.selectbox(
            "Family History",
            ["No", "Yes"]
        )

    if st.button("🔍 Predict Risk"):

        input_data = pd.DataFrame({

            "Age": [age],
            "Glucose": [glucose],
            "Systolic_BP": [bp],
            "Cholesterol": [chol],
            "BMI": [bmi],
            "Heart_Rate": [heart_rate],
            "Gender": [gender],
            "Smoking": [smoking],
            "Diabetes": [diabetes],
            "Family_History": [family_history]

        })

        input_data = pd.get_dummies(input_data)

        for col in features:
            if col not in input_data.columns:
                input_data[col] = 0

        input_data = input_data[features]

        scaled = scaler.transform(input_data)

        prediction = model.predict(scaled)[0]

        probability = model.predict_proba(
            scaled
        )[0][1]

        st.session_state["risk"] = prediction
        st.session_state["prob"] = probability
        st.session_state["history"].append({
            "Age": age,
            "Gender": gender,
            "Risk Score": round(probability * 100, 2)
        })

        st.markdown("---")

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                f"🔴 HIGH RISK ({probability:.1%})"
            )

        else:

            st.success(
                f"🟢 LOW RISK ({probability:.1%})"
            )

        st.metric(
            "Risk Probability",
            f"{probability:.1%}"
        )

        # [BUG FIX #6] — removed duplicate st.progress() call; keeping only one under Risk Meter
        st.subheader("Risk Meter")

        st.progress(float(probability))

        if probability > 0.7:

            st.error("High Risk Zone")

        elif probability > 0.4:

            st.warning("Moderate Risk Zone")

        else:

            st.success("Low Risk Zone")

# ==========================
# DOCTOR VIEW
# ==========================

elif page == "Doctor View":

    st.title("👨‍⚕️ Doctor Assessment Panel")

    if "risk" not in st.session_state:

        st.info(
            "Please run a prediction in Patient View first."
        )

    else:

        probability = st.session_state["prob"]

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Risk Probability",
                f"{probability:.1%}"
            )

        with c2:

            if probability > 0.70:

                st.error(
                    "High Cardiovascular Risk"
                )

            elif probability > 0.40:

                st.warning(
                    "Moderate Risk"
                )

            else:

                st.success(
                    "Low Risk"
                )

        st.markdown("---")

        st.subheader(
            "Clinical Recommendations"
        )

        if probability > 0.70:

            st.error("""
Immediate clinical evaluation recommended.
""")

            st.write("• Cardiology consultation")
            st.write("• Blood pressure monitoring")
            st.write("• Cholesterol management")
            st.write("• Smoking cessation")
            st.write("• Structured exercise plan")

        elif probability > 0.40:

            st.warning("""
Lifestyle modifications strongly advised.
""")

            st.write("• Increase physical activity")
            st.write("• Improve diet quality")
            st.write("• Reduce processed foods")
            st.write("• Monitor blood pressure")

        else:

            st.success("""
Current risk level appears low.
""")

            st.write("• Continue healthy lifestyle")
            st.write("• Annual health screening")
            st.write("• Maintain ideal body weight")

# ==========================
# ANALYTICS
# ==========================

elif page == "Analytics":

    st.title("📊 Analytics Dashboard")

    df = pd.read_csv("heart.csv")

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:

        st.subheader("Age Distribution")

        st.bar_chart(df["Age"])

    with row1_col2:

        st.subheader("BMI Distribution")

        st.bar_chart(df["BMI"])

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:

        st.subheader("Glucose Levels")

        st.line_chart(df["Glucose"])

    with row2_col2:

        st.subheader("Cholesterol Levels")

        st.line_chart(df["Cholesterol"])

    st.markdown("---")

    st.subheader("Dataset Statistics")

    st.dataframe(
        df.describe()
    )

    st.markdown("---")

    # [BUG FIX #1 & #2] — moved feature importance block inside the elif Analytics block
    st.subheader("Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    st.bar_chart(
        importance_df.set_index("Feature")
    )

# ==========================
# BULK PREDICTION
# ==========================

elif page == "Bulk Prediction":

    st.title("📂 Bulk Patient Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        patient_df = pd.read_csv(
            uploaded_file
        )

        st.success(
            f"{len(patient_df)} records uploaded successfully"
        )

        st.subheader(
            "Uploaded Dataset"
        )

        st.dataframe(patient_df)

        prediction_df = patient_df.copy()

        transformed_df = pd.get_dummies(
            patient_df
        )

        for col in features:

            if col not in transformed_df.columns:

                transformed_df[col] = 0

        transformed_df = transformed_df[
            features
        ]

        scaled = scaler.transform(
            transformed_df
        )

        predictions = model.predict(
            scaled
        )

        probabilities = model.predict_proba(
            scaled
        )[:, 1]

        prediction_df["Risk Category"] = [

            "High Risk" if prob > 0.70

            else "Moderate Risk" if prob > 0.40

            else "Low Risk"

            for prob in probabilities
        ]

        prediction_df["Risk Score"] = (
            probabilities * 100
        ).round(2)

        st.subheader(
            "Prediction Results"
        )

        st.dataframe(
            prediction_df
        )

        csv = prediction_df.to_csv(
            index=False
        )

        st.download_button(
            label="⬇ Download Results",
            data=csv,
            file_name="heart_risk_predictions.csv",
            mime="text/csv"
        )

# ==========================
# PREDICTION HISTORY        [BUG FIX #3] — added missing page
# ==========================

elif page == "Prediction History":

    st.title("📋 Prediction History")

    if not st.session_state["history"]:

        st.info(
            "No predictions made yet. Use Patient View to get started."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state["history"]
        )

        st.dataframe(history_df)

        st.metric(
            "Total Predictions Made",
            len(history_df)
        )

        if st.button("🗑️ Clear History"):
            st.session_state["history"] = []
            st.rerun()

# ==========================
# MODEL INFO                [BUG FIX #4] — added missing page
# ==========================

elif page == "Model Info":

    st.title("🤖 Model Information")

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.metric("Model Type", type(model).__name__)
        st.metric("Number of Features", len(features))

    with c2:
        st.metric("Scaler Type", type(scaler).__name__)

    st.markdown("---")

    st.subheader("Features Used")

    st.write(features)

# ==========================
# ABOUT PROJECT             [BUG FIX #4] — added missing page
# ==========================

elif page == "About Project":

    st.title("ℹ️ About This Project")

    st.markdown("---")

    st.info("""
**Heart Disease Risk Prediction System**

This application uses a machine learning model trained on clinical
patient data to predict cardiovascular disease risk.

**Key Features:**
- Single patient risk assessment (Patient View)
- Doctor recommendations based on risk score (Doctor View)
- Dataset analytics and visualizations (Analytics)
- Bulk CSV prediction for multiple patients (Bulk Prediction)
- Prediction history tracking

**Tech Stack:**
- Python · Streamlit · Scikit-learn · Pandas · Matplotlib
""")

    st.markdown("---")

    st.subheader("Disclaimer")

    st.warning("""
This tool is intended for educational and research purposes only.
It is NOT a substitute for professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare provider.
""")