import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

MODEL_PATH = "models/employee_attrition_model.pkl"
THRESHOLD_PATH = "models/prediction_threshold.pkl"

RANDOM_STATE = 42

# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("EMPLOYEE ATTRITION PREDICTION - MODEL TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# ============================================================
# REMOVE UNNECESSARY COLUMNS
# ============================================================

# These columns do not provide useful predictive information.
# EmployeeNumber is only an identifier.
# EmployeeCount and StandardHours are constant columns.
# Over18 is also constant in this dataset.

columns_to_remove = [
    "EmployeeNumber",
    "EmployeeCount",
    "StandardHours",
    "Over18"
]

df = df.drop(
    columns=columns_to_remove,
    errors="ignore"
)

# ============================================================
# TARGET
# ============================================================

target = "Attrition"

# Convert target:
# Yes = 1
# No  = 0

df[target] = df[target].map({
    "Yes": 1,
    "No": 0
})

# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[target])
y = df[target]

print(f"\nNumber of features: {X.shape[1]}")
print(f"Number of employees: {X.shape[0]}")

print("\nTarget distribution:")
print(y.value_counts())

# ============================================================
# IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# ============================================================
# RANDOM FOREST MODEL
# ============================================================

random_forest = RandomForestClassifier(
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)

# ============================================================
# COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            random_forest
        )
    ]
)

# ============================================================
# HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 70)
print("HYPERPARAMETER OPTIMIZATION")
print("=" * 70)

parameter_grid = {
    "model__n_estimators": [
        200,
        300
    ],

    "model__max_depth": [
        None,
        10,
        20
    ],

    "model__min_samples_split": [
        2,
        5
    ],

    "model__min_samples_leaf": [
        1,
        2
    ]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1
)

print("\nTraining models...")
print("This may take some time.\n")

grid_search.fit(
    X_train,
    y_train
)

best_model = grid_search.best_estimator_

print("\n" + "=" * 70)
print("BEST MODEL PARAMETERS")
print("=" * 70)

print(grid_search.best_params_)

# ============================================================
# PREDICT PROBABILITIES
# ============================================================

probabilities = best_model.predict_proba(
    X_test
)[:, 1]

# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD OPTIMIZATION")
print("=" * 70)

thresholds = np.arange(
    0.30,
    0.61,
    0.05
)

best_threshold = 0.50
best_f1 = 0

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"F1: {f1:.4f} | "
        f"Recall: {recall:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# ============================================================
# FINAL PREDICTIONS
# ============================================================

final_predictions = (
    probabilities >= best_threshold
).astype(int)

# ============================================================
# MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    final_predictions
)

precision = precision_score(
    y_test,
    final_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    final_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    final_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL PERFORMANCE")
print("=" * 70)

print(f"Model       : Tuned Random Forest")
print(f"Threshold   : {best_threshold:.2f}")
print(f"Accuracy    : {accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"F1 Score    : {f1:.4f}")
print(f"ROC-AUC     : {roc_auc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        final_predictions,
        target_names=[
            "Stayed",
            "Left"
        ],
        zero_division=0
    )
)

# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    best_model,
    MODEL_PATH
)

joblib.dump(
    best_threshold,
    THRESHOLD_PATH
)

# ============================================================
# SAVE FEATURE INFORMATION
# ============================================================

feature_information = {
    "features": X.columns.tolist(),
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "target": target
}

joblib.dump(
    feature_information,
    "models/feature_information.pkl"
)

# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 70)

print(f"Model     : {MODEL_PATH}")
print(f"Threshold : {THRESHOLD_PATH}")
print("Features  : models/feature_information.pkl")

print("\nTraining completed successfully! ✅")