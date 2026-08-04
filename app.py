import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👨‍💼",
    layout="wide"
)

# =====================================================
# LOAD MODEL & DATA
# =====================================================

model = joblib.load("models/employee_attrition_model.pkl")
label_encoders = joblib.load("models/label_encoders.pkl")

employees = pd.read_csv("data/employees.csv")

# =====================================================
# LOAD LOGO
# =====================================================

logo = Image.open("assets/logo.png")

# =====================================================
# HEADER
# =====================================================

col1, col2 = st.columns([1, 4])

with col1:
    st.image(logo, width=150)

with col2:
    st.title("👨‍💼 Employee Attrition Predictor")
    st.write(
        "Predict whether an employee is likely to leave the company using a Machine Learning model."
    )

st.divider()

# =====================================================
# SIDEBAR - PREDICTION MODE
# =====================================================

st.sidebar.header("⚙️ Prediction Mode")

prediction_mode = st.sidebar.radio(
    "Select Prediction Type",
    [
        "Existing Employee",
        "Manual Entry"
    ]
)

st.sidebar.divider()
# =====================================================
# EXISTING EMPLOYEE MODE
# =====================================================

if prediction_mode == "Existing Employee":

    st.sidebar.header("👤 Existing Employee")

    employee_list = (
        employees["EmployeeID"] + " - " + employees["Name"]
    ).tolist()

    selected_employee = st.sidebar.selectbox(
        "Select Employee",
        employee_list
    )

    employee_id = selected_employee.split(" - ")[0]

    employee = employees[
        employees["EmployeeID"] == employee_id
    ].iloc[0]

    st.sidebar.success("Employee Loaded Successfully")

    age = int(employee["Age"])
    monthly_income = int(employee["MonthlyIncome"])
    distance_from_home = int(employee["DistanceFromHome"])
    total_working_years = int(employee["TotalWorkingYears"])
    years_at_company = int(employee["YearsAtCompany"])

    business_travel = employee["BusinessTravel"]
    department = employee["Department"]
    job_role = employee["JobRole"]
    gender = employee["Gender"]
    marital_status = employee["MaritalStatus"]
    overtime = employee["OverTime"]

    st.info(f"""
### 👤 Employee Profile

**Employee ID:** {employee["EmployeeID"]}

**Name:** {employee["Name"]}

**Department:** {department}

**Job Role:** {job_role}
""")

# =====================================================
# MANUAL ENTRY MODE
# =====================================================

else:

    st.sidebar.header("✍️ Manual Entry")

    age = st.sidebar.slider(
        "Age",
        18,
        60,
        35
    )

    monthly_income = st.sidebar.slider(
        "Monthly Income",
        1000,
        20000,
        5000,
        100
    )

    distance_from_home = st.sidebar.slider(
        "Distance From Home",
        1,
        30,
        5
    )

    total_working_years = st.sidebar.slider(
        "Total Working Years",
        0,
        40,
        10
    )

    years_at_company = st.sidebar.slider(
        "Years At Company",
        0,
        40,
        5
    )
    business_travel = st.sidebar.selectbox(
        "Business Travel",
        [
            "Non-Travel",
            "Travel_Rarely",
            "Travel_Frequently"
        ]
    )

    department = st.sidebar.selectbox(
        "Department",
        [
            "Sales",
            "Research & Development",
            "Human Resources"
        ]
    )

    job_role = st.sidebar.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

    gender = st.sidebar.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    marital_status = st.sidebar.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    overtime = st.sidebar.selectbox(
        "OverTime",
        [
            "No",
            "Yes"
        ]
    )

# =====================================================
# PREPARE INPUT DATA
# =====================================================

input_data = {
    "Age": age,
    "BusinessTravel": business_travel,
    "Department": department,
    "JobRole": job_role,
    "Gender": gender,
    "MaritalStatus": marital_status,
    "MonthlyIncome": monthly_income,
    "DistanceFromHome": distance_from_home,
    "TotalWorkingYears": total_working_years,
    "YearsAtCompany": years_at_company,
    "OverTime": overtime
}

encoded_data = input_data.copy()

for column in encoded_data:
    if column in label_encoders:
        encoded_data[column] = label_encoders[column].transform(
            [encoded_data[column]]
        )[0]

input_df = pd.DataFrame([encoded_data])

# =====================================================
# PREDICT BUTTON
# =====================================================

predict = st.button(
    "🚀 Predict Attrition",
    use_container_width=True
)
# =====================================================
# PREDICTION
# =====================================================

if predict:

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    stay_probability = probability[0] * 100
    leave_probability = probability[1] * 100

    st.divider()

    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.subheader("📊 Prediction Result")

    if prediction == 0:
        st.success("✅ Employee is Likely to Stay in the Company")
    else:
        st.error("⚠️ Employee is Likely to Leave the Company")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Stay Probability",
            value=f"{stay_probability:.2f}%"
        )
        st.progress(int(stay_probability))

    with col2:
        st.metric(
            label="Leave Probability",
            value=f"{leave_probability:.2f}%"
        )
        st.progress(int(leave_probability))

    # =====================================================
    # RISK LEVEL
    # =====================================================

    st.subheader("🚦 Employee Risk Level")

    if leave_probability < 30:
        st.success("🟢 LOW RISK")

    elif leave_probability < 60:
        st.warning("🟡 MEDIUM RISK")

    else:
        st.error("🔴 HIGH RISK")
            # =====================================================
    # EMPLOYEE SUMMARY
    # =====================================================

    st.subheader("📋 Employee Summary")

    if prediction_mode == "Existing Employee":
        st.info(
            f"""
👤 Employee : {employee["Name"]}

🏢 Department : {department}

💼 Job Role : {job_role}
"""
        )

    summary = pd.DataFrame(
        {
            "Feature": [
                "Age",
                "Business Travel",
                "Department",
                "Job Role",
                "Gender",
                "Marital Status",
                "Monthly Income",
                "Distance From Home",
                "Total Working Years",
                "Years At Company",
                "OverTime"
            ],
            "Value": [
                age,
                business_travel,
                department,
                job_role,
                gender,
                marital_status,
                monthly_income,
                distance_from_home,
                total_working_years,
                years_at_company,
                overtime
            ]
        }
    )

    st.table(summary)

    # =====================================================
    # HR RECOMMENDATION
    # =====================================================

    st.divider()

    st.subheader("💡 HR Recommendation")

    if prediction == 1:

        st.error("High Attrition Risk")

        st.markdown("""
### Recommended Actions

- Conduct one-to-one meetings with the employee.
- Review workload and overtime.
- Improve work-life balance.
- Review salary and employee benefits.
- Provide career growth opportunities.
- Increase employee engagement.
- Recognize employee achievements.
""")

    else:

        st.success("Low Attrition Risk")

        st.markdown("""
### Recommended Actions

- Continue employee engagement.
- Maintain work-life balance.
- Encourage career development.
- Provide learning and training opportunities.
- Recognize good performance.
- Continue regular performance reviews.
""")
        