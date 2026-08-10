import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CLEAN PROFESSIONAL CSS
# ============================================================

st.markdown(
    """
    <style>

    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #172033;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #667085;
        margin-bottom: 35px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 750;
        color: #172033;
        margin-top: 30px;
        margin-bottom: 18px;
    }

    .info-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px;
        min-height: 105px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .info-label {
        color: #667085;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .info-value {
        color: #172033;
        font-size: 23px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    files = [
        "WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "data/WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    ]

    for file in files:
        try:
            return pd.read_csv(file)
        except:
            pass

    return None


df = load_data()

if df is None:

    st.error(
        "Employee dataset was not found. "
        "Please keep WA_Fn-UseC_-HR-Employee-Attrition.csv "
        "in the same folder as app.py."
    )

    st.stop()


# ============================================================
# TRAIN MODEL
# ============================================================

@st.cache_resource
def train_model(data):

    data = data.copy()

    # Target
    y = data["Attrition"].map({
        "Yes": 1,
        "No": 0
    })

    # Remove unnecessary columns
    columns_to_remove = [
        "Attrition",
        "EmployeeCount",
        "EmployeeNumber",
        "Over18",
        "StandardHours"
    ]

    X = data.drop(
        columns=[
            c for c in columns_to_remove
            if c in data.columns
        ]
    )

    # Separate categorical and numerical features
    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns
            ),
            (
                "numerical",
                "passthrough",
                numerical_columns
            )
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return pipeline, accuracy


model, accuracy = train_model(df)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("⚙️ Prediction Settings")

    st.write(
        "Select an employee to analyze their "
        "workforce attrition risk."
    )

    st.divider()

    st.subheader("👤 Select Employee")

    employee_options = []

    for index, row in df.iterrows():

        employee_number = row.get(
            "EmployeeNumber",
            index + 1
        )

        employee_options.append(
            f"Employee #{index + 1} — ID EMP-{int(employee_number):04d}"
        )

    selected_employee = st.selectbox(
        "Employee",
        employee_options
    )

    selected_index = employee_options.index(
        selected_employee
    )

    employee = df.iloc[selected_index]

    st.success("✓ Employee record loaded")

    st.divider()

    st.metric(
        "Employees Available",
        f"{len(df):,}"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">👥 Employee Attrition Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine learning based workforce analytics for identifying '
    'employee attrition risk.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTION FOR CARDS
# ============================================================

def show_card(column, label, value):

    with column:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-label">{label}</div>
                <div class="info-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# EMPLOYEE PROFILE
# ============================================================

st.markdown(
    '<div class="section-title">👤 Employee Profile</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

employee_number = employee.get(
    "EmployeeNumber",
    selected_index + 1
)

show_card(
    col1,
    "Employee ID",
    f"EMP-{int(employee_number):04d}"
)

show_card(
    col2,
    "Age",
    int(employee["Age"])
)

show_card(
    col3,
    "Department",
    employee["Department"]
)

show_card(
    col4,
    "Job Role",
    employee["JobRole"]
)


# ============================================================
# EMPLOYMENT DETAILS
# ============================================================

st.markdown(
    '<div class="section-title">💼 Employment Details</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

show_card(
    col1,
    "Monthly Income",
    f"${int(employee['MonthlyIncome']):,}"
)

show_card(
    col2,
    "Job Level",
    int(employee["JobLevel"])
)

show_card(
    col3,
    "Years at Company",
    int(employee["YearsAtCompany"])
)

show_card(
    col4,
    "Years in Current Role",
    int(employee["YearsInCurrentRole"])
)


# ============================================================
# EMPLOYEE INSIGHTS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Employee Insights</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

show_card(
    col1,
    "Job Satisfaction",
    f"{int(employee['JobSatisfaction'])} / 4"
)

show_card(
    col2,
    "Job Involvement",
    f"{int(employee['JobInvolvement'])} / 4"
)

show_card(
    col3,
    "Work-Life Balance",
    f"{int(employee['WorkLifeBalance'])} / 4"
)

show_card(
    col4,
    "Environment Satisfaction",
    f"{int(employee['EnvironmentSatisfaction'])} / 4"
)


# ============================================================
# WORKLOAD & EXPERIENCE
# ============================================================

st.markdown(
    '<div class="section-title">⏱️ Workload & Experience</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

show_card(
    col1,
    "Overtime",
    employee["OverTime"]
)

show_card(
    col2,
    "Total Working Years",
    int(employee["TotalWorkingYears"])
)

show_card(
    col3,
    "Years With Manager",
    int(employee["YearsWithCurrManager"])
)

show_card(
    col4,
    "Training Times",
    int(employee["TrainingTimesLastYear"])
)


# ============================================================
# PREDICTION SECTION
# ============================================================

st.markdown(
    '<div class="section-title">🔮 Attrition Prediction</div>',
    unsafe_allow_html=True
)

st.write(
    "Use the trained Random Forest machine learning model "
    "to estimate this employee's attrition risk."
)

predict_button = st.button(
    "🚀 Predict Attrition Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    employee_input = employee.drop(
        labels=[
            "Attrition",
            "EmployeeCount",
            "EmployeeNumber",
            "Over18",
            "StandardHours"
        ],
        errors="ignore"
    )

    employee_input = pd.DataFrame(
        [employee_input]
    )

    probabilities = model.predict_proba(
        employee_input
    )[0]

    stay_probability = float(probabilities[0])
    leave_probability = float(probabilities[1])


    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if leave_probability >= 0.70:

        risk = "HIGH RISK"

        description = (
            "This employee shows several indicators "
            "associated with higher attrition risk. "
            "Management attention is recommended."
        )

    elif leave_probability >= 0.40:

        risk = "MEDIUM RISK"

        description = (
            "This employee shows moderate indicators "
            "of attrition risk. Preventive retention "
            "actions may be beneficial."
        )

    else:

        risk = "LOW RISK"

        description = (
            "This employee currently shows stronger "
            "retention indicators and a lower likelihood "
            "of leaving the organization."
        )


    # ========================================================
    # RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )

    if risk == "HIGH RISK":

        st.error(
            f"🔴 {risk}\n\n{description}"
        )

    elif risk == "MEDIUM RISK":

        st.warning(
            f"🟡 {risk}\n\n{description}"
        )

    else:

        st.success(
            f"🟢 {risk}\n\n{description}"
        )


    # ========================================================
    # PROBABILITY
    # ========================================================

    st.markdown(
        '<div class="section-title">📈 Risk Probability</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Likelihood to Stay",
            f"{stay_probability * 100:.1f}%"
        )

        st.progress(
            stay_probability
        )

    with col2:

        st.metric(
            "Likelihood to Leave",
            f"{leave_probability * 100:.1f}%"
        )

        st.progress(
            leave_probability
        )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    st.markdown(
        '<div class="section-title">🚦 Risk Level</div>',
        unsafe_allow_html=True
    )

    if risk == "HIGH RISK":

        st.error(
            "🔴 HIGH RISK — Employee requires closer "
            "attention and retention planning."
        )

    elif risk == "MEDIUM RISK":

        st.warning(
            "🟡 MEDIUM RISK — Preventive retention "
            "actions are recommended."
        )

    else:

        st.success(
            "🟢 LOW RISK — Employee currently shows "
            "stronger retention indicators."
        )


    # ========================================================
    # RECOMMENDED ACTIONS
    # ========================================================

    st.markdown(
        '<div class="section-title">💡 Recommended Actions</div>',
        unsafe_allow_html=True
    )

    if risk == "HIGH RISK":

        actions = [
            "Review overtime and workload pressure.",
            "Discuss career growth and promotion opportunities.",
            "Consider flexible work arrangements.",
            "Review compensation and recognition opportunities.",
            "Create a personalized employee retention plan."
        ]

    elif risk == "MEDIUM RISK":

        actions = [
            "Schedule a one-to-one employee check-in.",
            "Review workload and work-life balance.",
            "Explore training and career development.",
            "Recognize employee contributions.",
            "Monitor satisfaction and engagement regularly."
        ]

    else:

        actions = [
            "Continue supporting positive employee engagement.",
            "Maintain a healthy work-life balance.",
            "Encourage career development opportunities.",
            "Provide learning and training opportunities.",
            "Recognize strong performance regularly."
        ]

    for action in actions:

        st.write(
            f"• {action}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Employee Attrition Predictor • Machine Learning Workforce Analytics"
)