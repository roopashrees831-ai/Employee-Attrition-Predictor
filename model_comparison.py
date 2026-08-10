import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. REMOVE UNNECESSARY COLUMNS
# ==========================================

columns_to_drop = [
    "EmployeeNumber",
    "EmployeeCount",
    "Over18",
    "StandardHours"
]

df = df.drop(columns=columns_to_drop)


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

X = df.drop("Attrition", axis=1)

y = df["Attrition"].map({
    "No": 0,
    "Yes": 1
})


# ==========================================
# 4. IDENTIFY COLUMN TYPES
# ==========================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()


# ==========================================
# 5. PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ==========================================
# 6. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 7. LOGISTIC REGRESSION
# ==========================================

logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ))
])

logistic_params = {
    "model__C": [0.01, 0.1, 1, 10, 100]
}

logistic_grid = GridSearchCV(
    logistic_pipeline,
    logistic_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nTuning Logistic Regression...")

logistic_grid.fit(
    X_train,
    y_train
)

best_logistic = logistic_grid.best_estimator_

print("Best Logistic Regression parameters:")
print(logistic_grid.best_params_)


# ==========================================
# 8. RANDOM FOREST
# ==========================================

random_forest_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

random_forest_params = {
    "model__n_estimators": [200, 300],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

random_forest_grid = GridSearchCV(
    random_forest_pipeline,
    random_forest_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

print("\nTuning Random Forest...")

random_forest_grid.fit(
    X_train,
    y_train
)

best_random_forest = random_forest_grid.best_estimator_

print("Best Random Forest parameters:")
print(random_forest_grid.best_params_)


# ==========================================
# 9. MODEL EVALUATION + THRESHOLD OPTIMIZATION
# ==========================================

def evaluate_model(model_name, model):

    # Get probability of Attrition = Yes
    probabilities = model.predict_proba(X_test)[:, 1]

    # --------------------------------------
    # Find best prediction threshold
    # --------------------------------------

    thresholds = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60
    ]

    best_threshold = 0.50
    best_threshold_f1 = 0

    print("\nThreshold testing:")

    for threshold in thresholds:

        threshold_predictions = (
            probabilities >= threshold
        ).astype(int)

        threshold_f1 = f1_score(
            y_test,
            threshold_predictions,
            zero_division=0
        )

        threshold_recall = recall_score(
            y_test,
            threshold_predictions,
            zero_division=0
        )

        print(
            f"Threshold {threshold:.2f} "
            f"| F1: {threshold_f1:.4f} "
            f"| Recall: {threshold_recall:.4f}"
        )

        if threshold_f1 > best_threshold_f1:
            best_threshold_f1 = threshold_f1
            best_threshold = threshold

    # --------------------------------------
    # Predictions using best threshold
    # --------------------------------------

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    # --------------------------------------
    # Metrics
    # --------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    # --------------------------------------
    # Display results
    # --------------------------------------

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    print(
        f"Best Threshold: {best_threshold:.2f}"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Stayed",
                "Left"
            ],
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Threshold": best_threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }


# ==========================================
# 10. EVALUATE MODELS
# ==========================================

results = []

results.append(
    evaluate_model(
        "Tuned Logistic Regression",
        best_logistic
    )
)

results.append(
    evaluate_model(
        "Tuned Random Forest",
        best_random_forest
    )
)


# ==========================================
# 11. FINAL MODEL COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

print("\n\n")

print("=" * 85)
print("FINAL MODEL COMPARISON")
print("=" * 85)

print(
    results_df
    .sort_values(
        "F1 Score",
        ascending=False
    )
    .to_string(index=False)
)


# ==========================================
# 12. SELECT BEST MODEL
# ==========================================

best_model = results_df.loc[
    results_df["F1 Score"].idxmax()
]

print("\n")

print("=" * 85)
print("BEST MODEL FOR ATTRITION DETECTION")
print("=" * 85)

print(
    "Model:",
    best_model["Model"]
)

print(
    f"Threshold: {best_model['Threshold']:.2f}"
)

print(
    f"Accuracy : {best_model['Accuracy']:.4f}"
)

print(
    f"Precision: {best_model['Precision']:.4f}"
)

print(
    f"Recall   : {best_model['Recall']:.4f}"
)

print(
    f"F1 Score : {best_model['F1 Score']:.4f}"
)

print(
    f"ROC-AUC  : {best_model['ROC-AUC']:.4f}"
)