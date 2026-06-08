import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading Dataset...")

df = pd.read_csv("heart.csv")

print("Columns:")
print(df.columns.tolist())

# Create target column

df["Heart_Disease"] = (
    ((df["Age"] >= 60) & (df["Cholesterol"] >= 240))
    | ((df["Systolic_BP"] >= 140) & (df["Diabetes"] == "Yes"))
    | ((df["Glucose"] >= 140) & (df["Smoking"] == "Yes"))
    | ((df["BMI"] >= 30) & (df["Family_History"] == "Yes"))
).astype(int)

print("\nTarget Distribution:")
print(df["Heart_Disease"].value_counts())

# Features

X = df.drop(["Name", "Heart_Disease"], axis=1)

# Target

y = df["Heart_Disease"]

# Convert categorical values

X = pd.get_dummies(X, drop_first=True)

print("\nTraining Features:")
print(X.columns.tolist())

# Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# Model

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy:.2%}")

# Save files

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(X.columns.tolist(), "models/features.pkl")

print("\nModel Saved Successfully!")