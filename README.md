# Employee Attrition Predictor

## 🚀 Live Demo

👉 [Launch Employee Attrition Predictor](https://r-employee-attrition-predictor.streamlit.app/)

---

## 📌 Project Overview

Employee Attrition Predictor is a machine learning-based application designed to identify employees who may be at risk of leaving an organization.

The system analyzes employee information such as job satisfaction, work-life balance, overtime, income, job level, years at the company, and other workforce-related factors to predict employee attrition risk.

The application provides an interactive dashboard where an employee can be selected and analyzed using a trained machine learning model.

---

## 🎯 Problem Statement

Employee turnover can create significant challenges for organizations, including:

- Increased recruitment costs
- Loss of experienced employees
- Reduced productivity
- Increased workload for existing employees
- Difficulty in retaining skilled employees

This project aims to use machine learning to identify employees who may have a higher probability of leaving, allowing organizations to take preventive retention measures.

---

## 💡 Objectives

- Predict whether an employee is likely to leave the organization
- Identify employees with different levels of attrition risk
- Analyze factors associated with employee attrition
- Provide stay and leave probabilities
- Provide recommended actions for employee retention
- Present predictions through an easy-to-use web application

---

## 🤖 Machine Learning

The project uses a **Random Forest Classifier** for employee attrition prediction.

### Model Performance

- **Algorithm:** Random Forest
- **Dataset Size:** 1,470 employee records
- **Model Accuracy:** 83.7%

The project also includes model comparison functionality for evaluating different machine learning approaches.

---

## 📊 Features

### 👤 Employee Selection

Select an employee from the available employee records.

### 🧑‍💼 Employee Profile

Displays important employee information including:

- Employee ID
- Age
- Department
- Job Role

### 💼 Employment Details

Displays:

- Monthly Income
- Job Level
- Years at Company
- Years in Current Role

### 🔍 Employee Insights

Analyzes factors such as:

- Job Satisfaction
- Job Involvement
- Work-Life Balance
- Environment Satisfaction

### ⏱️ Workload & Experience

Displays workforce-related information such as:

- Overtime
- Total Working Years
- Years With Manager
- Training Times

### 🔮 Attrition Prediction

The system provides:

- Attrition risk level
- Likelihood to stay
- Likelihood to leave
- Recommended actions

### 🚦 Risk Categories

🟢 **Low Risk**

The employee currently shows stronger retention indicators.

🟡 **Medium Risk**

The employee shows some indicators that may require attention.

🔴 **High Risk**

The employee shows multiple indicators associated with higher attrition risk.

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Pickle
- Machine Learning
- Random Forest

---

## 📂 Project Structure

```text
Employee-Attrition-Predictor/
│
├── assets/
│   └── logo.png
│
├── models/
│   ├── employee_attrition_model.pkl
│   ├── feature_information.pkl
│   ├── label_encoders.pkl
│   └── prediction_threshold.pkl
│
├── app.py
├── model_comparison.py
├── train_model.py
├── requirements.txt
├── runtime.txt
├── README.md
└── WA_Fn-UseC_-HR-Employee-Attrition.csv
