# heart_disease_stretch_goals.py

# Required imports
import pandas as pd
import numpy as np
import pickle
import requests
import streamlit as st

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENWEATHER_API_KEY")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('heart.csv')
X = df.drop('target', axis=1)
y = df['target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Compare ML models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = acc

# Train final model and save it
final_model = RandomForestClassifier()
final_model.fit(X_train, y_train)
pickle.dump(final_model, open("model.pkl", "wb"))

# Weather API integration (OpenWeatherMap)
def fetch_weather(city):
    # Use the api_key loaded from environment or secrets
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json()
        return data['main']['temp'], data['main']['humidity']
    return None, None

# Streamlit app
st.title("Heart Disease Risk Predictor")

# Collect all 13 features
age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 400)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [1, 0])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 70, 220)
exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [1, 0])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, step=0.1)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1=Normal, 2=Fixed Defect, 3=Reversible Defect)", [1, 2, 3])

city = st.text_input("Your City (for real-time weather)")

if st.button("Predict"):
    model = pickle.load(open("model.pkl", "rb"))
    scaler = StandardScaler()
    scaler.fit(X)  # Fit scaler on the original data
    input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                            columns=X.columns)
    features_scaled = scaler.transform(input_df)
    prediction = model.predict(features_scaled)
    st.write("Heart Disease Risk:", "Yes" if prediction[0] else "No")

    if city:
        temp, humidity = fetch_weather(city)
        if temp is not None:
            st.write(f"Current Weather in {city}: {temp}°C, {humidity}% Humidity")
        else:
            st.write("Failed to fetch weather data.")

# Show model accuracy comparisons
st.subheader("Model Accuracy Comparison")
for model, acc in results.items():
    st.write(f"{model}: {acc:.2f}")
