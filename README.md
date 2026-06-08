# ❤️ Heart Disease Risk Prediction and Visualization

##  Overview

Heart Disease Risk Prediction and Visualization is a Machine Learning-powered web application that predicts an individual's risk of developing cardiovascular disease based on key clinical and lifestyle parameters.

The application provides:

- Patient Risk Assessment
- Doctor Decision Support
- Interactive Analytics Dashboard
- Bulk Patient Risk Prediction
- Lifestyle Recommendations
- Data Visualization

Built using Python, Machine Learning, and Streamlit.

---

##  Features

### 👤 Patient View
- Enter patient health parameters
- Predict heart disease risk instantly
- Risk probability score
- Risk meter visualization

### 👨‍⚕️ Doctor View
- Clinical risk assessment
- Risk categorization
- Personalized lifestyle recommendations

### 📊 Analytics Dashboard
- Age distribution analysis
- BMI analysis
- Cholesterol trends
- Glucose level visualization
- Feature importance visualization

### 📂 Bulk Prediction
- Upload CSV files
- Predict risk for multiple patients
- Download prediction results

### 📜 Prediction History
- Track previous predictions
- Review patient risk assessments

---

##  Machine Learning Model

### Algorithm
- Random Forest Classifier

### Data Preprocessing
- Data Cleaning
- One-Hot Encoding
- Feature Scaling (StandardScaler)

### Evaluation
- Train-Test Split
- Accuracy Measurement

---

##  Technology Stack

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-Learn
- Joblib
- Streamlit

### Visualization
- Streamlit Charts
- Interactive Dashboard Components

---

## 📁 Project Structure

```text
Heart-Disease-Risk-Prediction/
│
├── heart.csv
├── train_model.py
├── app.py
│
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── features.pkl
│
├── screenshots/
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/saishreshta-achu/Heart-disease-risk-prediction-and-visualization.git
```

### Navigate to Project

```bash
cd Heart-disease-risk-prediction-and-visualization
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Run Application

```bash
streamlit run app.py
```

---

##  Input Features

The model uses the following parameters:

- Gender
- Age
- Glucose
- Systolic Blood Pressure
- Cholesterol
- BMI
- Heart Rate
- Smoking Status
- Diabetes
- Family History

---

##  Risk Categories

| Probability | Risk Level |
|------------|------------|
| 0 - 40% | Low Risk |
| 40 - 70% | Moderate Risk |
| Above 70% | High Risk |



##  Future Enhancements

- ECG Integration
- SHAP Explainability
- LIME Interpretability
- PDF Medical Reports
- Doctor Authentication System
- Cloud Deployment
- Real-Time Patient Monitoring

---

##  Author

**Sai Shreshta Santhosh**

Computer Science Engineering Graduate

Skills:
- Python
- SQL
- Power BI
- Data Analytics
- Machine Learning
- Data Visualization

GitHub:
https://github.com/saishreshta-achu
