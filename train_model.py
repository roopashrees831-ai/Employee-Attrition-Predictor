import pandas as pd
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# ==============================
# Select Only Required Features
# ==============================

features = [
    "Age",
    "BusinessTravel",
    "Department",
    "JobRole",
    "Gender",
    "MaritalStatus",
    "MonthlyIncome",
    "DistanceFromHome",
    "TotalWorkingYears",
    "YearsAtCompany",
    "OverTime"
]

target = "Attrition"

df = df[features + [target]]

# ==============================
# Encode Categorical Columns
# ==============================

label_encoders = {}

categorical_columns = [
    "BusinessTravel",
    "Department",
    "JobRole",
    "Gender",
    "MaritalStatus",
    "OverTime",
    "Attrition"
]

for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# ==============================
# Split Data
# ==============================

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# Train Model
# ==============================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# Accuracy
# ==============================

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("=" * 50)
print(f"Model Accuracy : {accuracy*100:.2f}%")
print("=" * 50)

# ==============================
# Feature Importance
# ==============================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop Important Features\n")
print(importance)

# ==============================
# Save Model
# ==============================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/employee_attrition_model.pkl")
joblib.dump(label_encoders, "models/label_encoders.pkl")

print("\nModel Saved Successfully!")
print("Label Encoders Saved Successfully!")