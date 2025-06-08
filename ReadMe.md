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

## 🎯 SDG Impact

This project supports **SDG 3 - Good Health and Well-Being** by demonstrating how AI can enhance preventive healthcare and assist in timely diagnosis of chronic conditions.

---

## 📁 Project Structure

.
├── heart.csv # Dataset
├── main.py # Main script to run the model
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## 🚀 Getting Started

Follow these steps to run the project on your machine:

### 1. Clone the Repository

```bash
git clone https://github.com/ireneiroha/Predicting_Heart_Disease.git
cd Predicting_Heart_Disease

### 2. Install Required Libraries
```bash
pip install -r requirements.txt

### 3. Run the Project
```bash
heart_disease.py


