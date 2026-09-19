import os
import json
import pandas as pd
import numpy as np
import streamlit as st

from utils import (
    apply_custom_css, load_model_pipeline, load_metrics,
    render_metric_card
)

# Page Configuration
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Minimal & Clean CSS
apply_custom_css()

# Load Model Pipeline & Metrics
pipeline = load_model_pipeline()
metrics = load_metrics()

# Sidebar Navigation
st.sidebar.markdown("### 💼 HR Analytics")
st.sidebar.caption("Simple Attrition Risk Tool")
st.sidebar.divider()

nav_option = st.sidebar.radio(
    "Navigation Menu",
    ["Home", "Predict Attrition", "About Project"]
)

# ==========================================
# PAGE 1: HOME (CLEAN & SIMPLE)
# ==========================================
if nav_option == "Home":
    st.title("Employee Attrition Prediction")
    st.caption("A simple machine learning tool to assess employee retention risk.")
    st.divider()

    col1, col2, col3 = st.columns(3)
    if metrics:
        ds = metrics["dataset_info"]
        perf = metrics["performance"]
        with col1:
            render_metric_card("Total Dataset Records", f"{ds['total_records']:,}")
        with col2:
            render_metric_card("Model Accuracy", f"{perf['accuracy']*100:.1f}%")
        with col3:
            attr_rate = (ds['attrition_yes_count'] / ds['total_records']) * 100
            render_metric_card("Historical Attrition Rate", f"{attr_rate:.1f}%")

    st.markdown("""
        ### How It Works
        1. Go to **Predict Attrition** in the sidebar.
        2. Fill in the key employee details.
        3. Click **Predict Attrition** to get instant retention probability and risk level.
    """)


# ==========================================
# PAGE 2: PREDICT ATTRITION (ULTRA-SIMPLE UI)
# ==========================================
elif nav_option == "Predict Attrition":
    st.title("Predict Employee Attrition")
    st.caption("Fill in key employee details to calculate retention risk.")
    st.divider()

    if pipeline is None:
        st.error("⚠️ Model pipeline file (`employee_attrition_pipeline.pkl`) not found. Please run `train_model.py`.")
    else:
        with st.form("simple_attrition_form"):
            st.markdown("#### 🔑 Key Employee Attributes")
            
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Age", min_value=18, max_value=65, value=35)
                department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
                job_role = st.selectbox("Job Role", [
                    "Sales Executive", "Research Scientist", "Laboratory Technician",
                    "Manufacturing Director", "Healthcare Representative", "Manager",
                    "Sales Representative", "Research Director", "Human Resources"
                ])
                monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=5000, step=500)
                overtime = st.selectbox("OverTime Requirement", ["No", "Yes"])

            with c2:
                distance_from_home = st.number_input("Distance From Home (miles/km)", min_value=1, max_value=30, value=5)
                job_satisfaction = st.select_slider("Job Satisfaction (1-Low to 4-High)", options=[1, 2, 3, 4], value=3)
                work_life_balance = st.select_slider("Work-Life Balance (1-Low to 4-High)", options=[1, 2, 3, 4], value=3)
                total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=8)
                years_at_company = st.number_input("Years At Company", min_value=0, max_value=40, value=5)

            # Collapsible Advanced Fields with Smart Defaults
            with st.expander("⚙️ Additional Attributes (Optional / Defaults Pre-filled)"):
                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    gender = st.selectbox("Gender", ["Female", "Male"])
                    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], index=1)
                    business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
                    education = st.selectbox("Education Level", [1, 2, 3, 4, 5], index=2)
                    education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
                    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5], index=1)
                    job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4], index=2)
                with ac2:
                    env_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4], index=2)
                    relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4], index=2)
                    performance_rating = st.selectbox("Performance Rating", [3, 4], index=0)
                    daily_rate = st.number_input("Daily Rate ($)", min_value=100, max_value=1500, value=800)
                    hourly_rate = st.number_input("Hourly Rate ($)", min_value=30, max_value=100, value=65)
                    monthly_rate = st.number_input("Monthly Rate ($)", min_value=2000, max_value=30000, value=14000)
                    percent_salary_hike = st.slider("Percent Salary Hike (%)", min_value=11, max_value=25, value=14)
                with ac3:
                    stock_option_level = st.selectbox("Stock Option Level", [0, 1, 2, 3], index=1)
                    num_companies_worked = st.number_input("Num Companies Worked", min_value=0, max_value=9, value=2)
                    training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=3)
                    years_in_current_role = st.number_input("Years In Current Role", min_value=0, max_value=20, value=3)
                    years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1)
                    years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=20, value=3)

            submit_btn = st.form_submit_button("🚀 Predict Attrition", use_container_width=True)

        if submit_btn:
            input_dict = {
                'Age': age, 'BusinessTravel': business_travel, 'DailyRate': daily_rate,
                'Department': department, 'DistanceFromHome': distance_from_home,
                'Education': education, 'EducationField': education_field,
                'EnvironmentSatisfaction': env_satisfaction, 'Gender': gender,
                'HourlyRate': hourly_rate, 'JobInvolvement': job_involvement,
                'JobLevel': job_level, 'JobRole': job_role,
                'JobSatisfaction': job_satisfaction, 'MaritalStatus': marital_status,
                'MonthlyIncome': monthly_income, 'MonthlyRate': monthly_rate,
                'NumCompaniesWorked': num_companies_worked, 'OverTime': overtime,
                'PercentSalaryHike': percent_salary_hike, 'PerformanceRating': performance_rating,
                'RelationshipSatisfaction': relationship_satisfaction,
                'StockOptionLevel': stock_option_level, 'TotalWorkingYears': total_working_years,
                'TrainingTimesLastYear': training_times_last_year,
                'WorkLifeBalance': work_life_balance, 'YearsAtCompany': years_at_company,
                'YearsInCurrentRole': years_in_current_role,
                'YearsSinceLastPromotion': years_since_last_promotion,
                'YearsWithCurrManager': years_with_curr_manager
            }
            
            input_df = pd.DataFrame([input_dict])

            try:
                prediction_class = pipeline.predict(input_df)[0]
                probabilities = pipeline.predict_proba(input_df)[0]
                
                stay_prob = probabilities[0] * 100
                leave_prob = probabilities[1] * 100

                st.divider()
                
                if leave_prob < 30.0:
                    st.success(f"✅ **Prediction: Likely to Stay** (Risk Level: Low | Leave Probability: {leave_prob:.1f}%)")
                elif leave_prob <= 60.0:
                    st.warning(f"⚠️ **Prediction: Moderate Retention Risk** (Risk Level: Medium | Leave Probability: {leave_prob:.1f}%)")
                else:
                    st.error(f"🚨 **Prediction: Likely to Leave** (Risk Level: High | Leave Probability: {leave_prob:.1f}%)")

                st.write(f"**Stay Probability:** {stay_prob:.1f}%")
                st.progress(int(stay_prob))

                st.write(f"**Leave Probability:** {leave_prob:.1f}%")
                st.progress(int(leave_prob))

            except Exception as e:
                st.error(f"Prediction Error: {e}")


# ==========================================
# PAGE 3: ABOUT PROJECT
# ==========================================
elif nav_option == "About Project":
    st.title("About Project")
    st.divider()

    st.markdown("""
        **Employee Attrition Prediction App**
        - Built using **Python**, **Scikit-Learn**, **Pandas**, and **Streamlit**.
        - Powered by a real **Random Forest Classification Pipeline**.
        - Trained on the official IBM HR Analytics Employee Attrition & Performance dataset.
    """)
