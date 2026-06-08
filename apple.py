import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Patient View",
        "Doctor View",
        "Analytics",
        "Upload Dataset"
    ]
)
if page == "Dashboard":

    st.title("❤️ Heart Disease Risk Prediction")

    df = pd.read_csv("heart.csv")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Patients", len(df))

    col2.metric(
        "Average Age",
        round(df["Age"].mean(), 1)
    )

    col3.metric(
        "Average Cholesterol",
        round(df["Cholesterol"].mean(), 1)
    )

    col4.metric(
        "Average BMI",
        round(df["BMI"].mean(), 1)
    )

    st.dataframe(df.head())
elif page == "Patient View":

    st.title("Patient Risk Assessment")

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
    if st.button("Predict Risk"):

        input_data = pd.DataFrame({

            "Age":[age],
            "Glucose":[glucose],
            "Systolic_BP":[bp],
            "Cholesterol":[chol],
            "BMI":[bmi],
            "Heart_Rate":[heart_rate],
            "Gender":[gender],
            "Smoking":[smoking],
            "Diabetes":[diabetes],
            "Family_History":[family_history]

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

        if prediction == 1:

            st.error(
                f"High Risk ({probability:.1%})"
            )

        else:

            st.success(
                f"Low Risk ({probability:.1%})"
            )
elif page == "Doctor View":

    st.title("Doctor Assessment")

    if "risk" not in st.session_state:

        st.info(
            "Run Patient View first"
        )

    else:

        probability = st.session_state["prob"]

        st.metric(
            "Risk Probability",
            f"{probability:.1%}"
        )

        if probability > 0.70:

            st.warning(
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
        st.subheader(
            "Lifestyle Recommendations"
        )

        st.write(
            "• Exercise 30 minutes daily"
        )

        st.write(
            "• Maintain healthy weight"
        )

        st.write(
            "• Monitor cholesterol"
        )

        st.write(
            "• Reduce salt intake"
        )

        st.write(
            "• Avoid smoking"
        )
elif page == "Analytics":

    st.title("Analytics")

    df = pd.read_csv("heart.csv")

    st.subheader(
        "Age Distribution"
    )

    st.bar_chart(df["Age"])

    st.subheader(
        "Cholesterol Distribution"
    )

    st.bar_chart(
        df["Cholesterol"]
    )

    st.subheader(
        "BMI Distribution"
    )

    st.bar_chart(
        df["BMI"]
    )
elif page == "Upload Dataset":

    st.title("Bulk Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file:

        patient_df = pd.read_csv(
            uploaded_file
        )

        st.write(
            "Uploaded Dataset"
        )

        st.dataframe(
            patient_df.head()
        )
elif page == "Upload Dataset":

    st.title("Bulk Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file:

        patient_df = pd.read_csv(
            uploaded_file
        )

        st.write(
            "Uploaded Dataset"
        )

        st.dataframe(
            patient_df.head()
        )
        prediction_df = patient_df.copy()

        patient_df = pd.get_dummies(
            patient_df
        )

        for col in features:

            if col not in patient_df.columns:

                patient_df[col] = 0

        patient_df = patient_df[
            features
        ]

        scaled = scaler.transform(
            patient_df
        )

        results = model.predict(
            scaled
        )

        prediction_df[
            "Prediction"
        ] = results

        st.dataframe(
            prediction_df
        )

        st.download_button(
            "Download Results",
            prediction_df.to_csv(
                index=False
            ),
            "predictions.csv"
        )
st.set_page_config(
    page_title="Heart Disease Risk Prediction",
    page_icon="❤️",
    layout="wide"
)
st.markdown("""
<style>

/* Main Background */
.stApp{
    background: linear-gradient(
        135deg,
        #000000,
        #0f0f0f,
        #1a0000
    );
    color:white;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background-color:#120000;
    border-right:2px solid #ff4b4b;
}

/* Titles */
h1,h2,h3{
    color:#ff4b4b !important;
}

/* Metric Cards */
div[data-testid="metric-container"]{
    background:#1c1c1c;
    border:1px solid #ff4b4b;
    padding:20px;
    border-radius:15px;
    box-shadow:0 0 15px rgba(255,75,75,0.4);
}

/* Buttons */
.stButton button{
    background:#ff4b4b;
    color:white;
    border:none;
    border-radius:10px;
    font-weight:bold;
}

.stButton button:hover{
    background:#ff1a1a;
}

/* Input Boxes */
.stNumberInput,
.stSelectbox{
    background:#1c1c1c;
}

/* Dataframes */
[data-testid="stDataFrame"]{
    border:1px solid #ff4b4b;
    border-radius:12px;
}

/* Success */
.stSuccess{
    border-left:5px solid #00ff88;
}

/* Error */
.stError{
    border-left:5px solid #ff4b4b;
}

/* Warning */
.stWarning{
    border-left:5px solid orange;
}

</style>
""", unsafe_allow_html=True)