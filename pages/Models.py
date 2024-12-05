# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostClassifier

# Gautami
df = pd.read_csv("fetal_health_gautami.csv")
# Importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import f_classif, SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from scipy.stats.mstats import winsorize

# Splitting data into features (X) and target (y)
X = df.drop(columns=["fetal_health"])
y = df["fetal_health"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Evaluating models and storing results
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Sorting models by accuracy
sorted_results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

# Streamlit title and description
st.title("Fetal Health Prediction")
st.write("## Model Comparison by Accuracy")
st.write("This chart compares the accuracy of different models used for fetal health prediction.")

# Plotting the chart
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=list(sorted_results.keys()), y=list(sorted_results.values()), palette="viridis", ax=ax)
ax.set_title("Model Comparison by Accuracy", fontsize=16)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_xlabel("Model", fontsize=12)
ax.set_xticklabels(list(sorted_results.keys()), rotation=45, fontsize=10)
ax.set_ylim(0, 1)

# Add accuracy values on top of each bar
for i, value in enumerate(sorted_results.values()):
    ax.text(i, value + 0.02, f"{value:.2f}", ha='center', va='bottom', fontsize=10, color='black')

fig.tight_layout()
st.pyplot(fig)

# Feature selection
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

# Numerical features: F-test
f_test = SelectKBest(score_func=f_classif, k='all')
f_test.fit(X[numerical_cols], y)
anova_scores = pd.DataFrame({'Feature': numerical_cols, 'F-Score': f_test.scores_})

# Categorical features: Chi-Square test
if len(categorical_cols) > 0:
    X_categorical = X[categorical_cols].apply(LabelEncoder().fit_transform)
    scaler = MinMaxScaler()
    X_categorical_scaled = scaler.fit_transform(X_categorical)
    chi2_test = SelectKBest(score_func=chi2, k='all')
    chi2_test.fit(X_categorical_scaled, y)
    chi2_scores = pd.DataFrame({'Feature': categorical_cols, 'Chi-Square': chi2_test.scores_})
else:
    chi2_scores = pd.DataFrame({'Feature': [], 'Chi-Square': []})

# Combining feature scores
feature_scores = pd.concat([anova_scores, chi2_scores], axis=0, ignore_index=True)

# Random Forest feature importance
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X, y)
rf_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Streamlit Feature Importance
st.title("Random Forest Feature Importance")
st.subheader("Feature Importance Scores")
st.write(rf_importances)

# Top 10 feature importances
top_features = rf_importances.head(10)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=top_features, palette="coolwarm", ax=ax)
ax.set_title("Top 10 Feature Importances - Random Forest", fontsize=16)
ax.set_xlabel("Importance", fontsize=12)
ax.set_ylabel("Features", fontsize=12)
plt.tight_layout()
st.pyplot(fig)

# Accuracy Comparison for Maternal Health Models
data = pd.read_csv('Expanded_Maternal_Health_Risk_Data.csv')
data.drop(columns=['SystolicBP'], inplace=True)
data.drop(index=data[data['HeartRate'] == 7].index, inplace=True)

X = data[["Age", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]]
y = data["RiskLevel"]
y = y.map({"low risk": 0, "mid risk": 1, "high risk": 2})

accuracy_maternal = {}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
lr_model = LogisticRegression(
    C=0.01,
    max_iter=100,
    solver='liblinear',
    multi_class='ovr'
)
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
accuracy_maternal['LogisticRegression'] = accuracy_score(y_test, y_pred)

# KNN
knn_model = KNeighborsClassifier(n_neighbors=10, weights='distance')
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
accuracy_maternal['KNN'] = accuracy_score(y_test, y_pred)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=400, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_maternal['RandomForest'] = accuracy_score(y_test, y_pred)

# CatBoost
catboost_model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, verbose=False)
catboost_model.fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)
accuracy_maternal['Catboost'] = accuracy_score(y_test, y_pred)

# Accuracy plot
plt.figure(figsize=(10, 6))
plt.bar(accuracy_maternal.keys(), accuracy_maternal.values(), color=['skyblue', 'orange', 'green', 'red'])
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Maternal Health Model Accuracies', fontsize=14)
plt.tight_layout()
st.title("Maternal Health Model Accuracy Comparison")
st.pyplot(plt)
