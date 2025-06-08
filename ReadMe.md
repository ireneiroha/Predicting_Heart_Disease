# 🌍 SDG 3 - Heart Disease Prediction using Machine Learning

This project supports **Sustainable Development Goal 3: Good Health and Well-Being** by using machine learning to **predict the risk of heart disease** based on patient data.

## 🧠 Problem Statement

Heart disease is one of the leading causes of death globally. Early detection can save lives and reduce long-term healthcare costs. This project applies a supervised learning approach to classify whether a patient is at risk of heart disease.

---

## 📊 Dataset

- **Source**: [Kaggle Heart Disease Dataset](https://www.kaggle.com/)
- **File Used**: `heart.csv`
- **Features** include:
  - `age`, `sex`, `cp` (chest pain type), `trestbps` (resting blood pressure)
  - `chol` (cholesterol), `fbs` (fasting blood sugar), `thalach` (max heart rate), etc.
- **Target**: `target` (1 = disease present, 0 = no disease)

---

## 🛠️ Tools and Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 🚀 ML Approach

- **Type**: Supervised Learning – Classification
- **Model Used**: Random Forest Classifier
- **Steps**:
  1. Data Cleaning & Preprocessing
  2. Feature Scaling using StandardScaler
  3. Train-test split (80/20)
  4. Model Training and Evaluation

---

## 📈 Results

- The model achieved **high accuracy** with balanced precision and recall.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score
- **Confusion Matrix** visualized with Seaborn.

---

## 🤖 Ethical & Social Considerations

- **Bias**: Dataset may not represent all populations (e.g., by age, race).
- **Fairness**: The model is an assistive tool, not a replacement for medical professionals.
- **Sustainability**: Helps prioritize early intervention, reducing healthcare burden in the long term.

---

## 🌐 Deployment and Stretch Goals

### 🚪 Web App Deployment
- The model is deployed as a **Streamlit web application**.
- Users can enter patient data through a web form to receive real-time predictions.
- The app can be hosted publicly via **Streamlit Cloud**.

### 📦 Model Comparison
- Multiple ML models were trained and evaluated:
  - Random Forest, Logistic Regression, SVM, Gradient Boosting
- Random Forest and SVM achieved the best performance (~87% accuracy).

### ✨ Real-Time API Integration
- Integrated **OpenWeatherMap API** to fetch current weather data (e.g., temperature and humidity).
- Demonstrates how real-world environmental conditions can be layered into ML-based health predictions.


