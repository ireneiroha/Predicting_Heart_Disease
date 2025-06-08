# 1. Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                            classification_report, roc_auc_score)

# Set up visualization
%matplotlib inline
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

# 2. Load and inspect data
print("Loading and exploring data...")
df = pd.read_csv('heart.csv')
print(f"\nData shape: {df.shape}")

print("\nFirst 5 rows:")
display(df.head())

print("\nData summary:")
print(df.describe())

print("\nMissing values check:")
print(df.isnull().sum())

# 3. Data visualization
print("\nVisualizing data distributions...")
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()

# Target distribution
plt.figure()
sns.countplot(x='target', data=df)
plt.title('Heart Disease Distribution (0 = No, 1 = Yes)')
plt.show()

# 4. Prepare data for modeling
print("\nPreparing data for modeling...")
X = df.drop('target', axis=1)
y = df['target']

# Split data 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================
# 5. LOGISTIC REGRESSION MODEL 
# ==============================================
print("\n" + "="*50)
print("LOGISTIC REGRESSION MODEL")
print("="*50)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

print("\nLogistic Regression Performance:")
print(f"Accuracy: {accuracy_score(y_test, lr_pred):.2f}")
print(f"AUC Score: {roc_auc_score(y_test, lr_pred):.2f}")

print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

# Confusion Matrix
plt.figure()
sns.heatmap(confusion_matrix(y_test, lr_pred), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title('Logistic Regression Confusion Matrix')
plt.show()

# ==============================================
# 6. RANDOM FOREST MODEL (Your partner's approach)
# ==============================================
print("\n" + "="*50)
print("RANDOM FOREST MODEL")
print("="*50)

# Note: Random Forest doesn't require feature scaling
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)  # Using unscaled data
rf_pred = rf.predict(X_test)

print("\nRandom Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.2f}")
print(f"AUC Score: {roc_auc_score(y_test, rf_pred):.2f}")

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# Confusion Matrix
plt.figure()
sns.heatmap(confusion_matrix(y_test, rf_pred), 
            annot=True, fmt='d', cmap='Reds',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'])
plt.title('Random Forest Confusion Matrix')
plt.show()

# Feature Importance
plt.figure()
feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features (Random Forest)')
plt.show()

# ==============================================
# 7. MODEL COMPARISON
# ==============================================
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, rf_pred)
    ],
    'AUC Score': [
        roc_auc_score(y_test, lr_pred),
        roc_auc_score(y_test, rf_pred)
    ]
})

print("\nPerformance Comparison:")
display(results)

# ==============================================
# 8. MAKING PREDICTIONS
# ==============================================
print("\n" + "="*50)
print("SAMPLE PREDICTION")
print("="*50)

# Using first patient from test set as example
sample_data = X_test.iloc[0:1]
sample_scaled = scaler.transform(sample_data)

# Logistic Regression prediction
lr_prob = lr.predict_proba(sample_scaled)[0][1]
print(f"\nLogistic Regression Prediction:")
print(f"Probability of heart disease: {lr_prob:.1%}")
print(f"Predicted class: {'Disease' if lr.predict(sample_scaled)[0] else 'No Disease'}")

# Random Forest prediction
rf_prob = rf.predict_proba(sample_data)[0][1]
print(f"\nRandom Forest Prediction:")
print(f"Probability of heart disease: {rf_prob:.1%}")
print(f"Predicted class: {'Disease' if rf.predict(sample_data)[0] else 'No Disease'}")

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
