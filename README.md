# Employee Attrition Prediction Machine Learning Web Application

> **Machine Learning Based Employee Retention Risk Analysis**

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

---

## 📌 Project Overview
**Employee Attrition Prediction** is an end-to-end Machine Learning web application designed to help HR departments and managers evaluate employee turnover risk. Using the IBM HR Analytics Employee Attrition dataset, the system trains a Random Forest Classifier within a scikit-learn preprocessing pipeline to predict whether an employee is likely to leave (**Attrition = Yes**) or stay (**Attrition = No**), providing clear probability estimates and risk levels (Low, Medium, High).

---

## 🎯 Problem Statement
Unplanned employee departure causes financial loss, project delays, and team morale disruption. Identifying key drivers behind turnover (such as overtime work, salary hikes, commute distance, and job satisfaction) enables organizations to implement proactive retention strategies before valuable talent leaves.

---

## 📊 Dataset Information
- **Dataset:** IBM HR Analytics Employee Attrition & Performance Dataset (`WA_Fn-UseC_-HR-Employee-Attrition.csv`)
- **Records:** 1,470 employee entries
- **Attributes:** 35 columns (demographics, job roles, satisfaction levels, compensation, career history)
- **Target Variable:** `Attrition` ('Yes' / 'No')
  - **No (Stay):** 1,233 (83.9%)
  - **Yes (Leave):** 237 (16.1%)

---

## 🛠️ Technology Stack
- **Language:** Python 3.12
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (RandomForestClassifier, OneHotEncoder, StandardScaler, ColumnTransformer, Pipeline)
- **Model Serialization:** Joblib
- **Web Interface:** Streamlit (Custom CSS, responsive multi-page dashboard)
- **Data Visualization:** Plotly, Matplotlib

---

## 🤖 ML Algorithm & Architecture
The predictive core uses a Scikit-Learn `Pipeline`:
1. **Feature Cleanup:** Constant columns (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`) are automatically removed.
2. **ColumnTransformer:**
   - **Categorical Columns (7):** `BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime` encoded via `OneHotEncoder(handle_unknown='ignore')`.
   - **Numerical Columns (23):** Scaled using `StandardScaler()`.
3. **Classifier:** `RandomForestClassifier(random_state=42, class_weight='balanced')` to handle the imbalanced attrition dataset.

---

## 📈 Model Evaluation
Trained and evaluated on an 80% Train / 20% Stratified Test split (294 test samples):

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **83.67%** |
| **Precision (Stay - No)** | **88.42%** |
| **Recall (Stay - No)** | **92.71%** |
| **F1-Score (Stay - No)** | **90.51%** |
| **Precision (Leave - Yes)** | **48.57%** |
| **Recall (Leave - Yes)** | **36.17%** |
| **F1-Score (Leave - Yes)** | **41.46%** |

---

## 🌟 Key Application Features
1. **Home Page:** Executive dashboard with key project stats, attrition rate, and ML algorithm overview.
2. **Predict Attrition:** Interactive input form organized by domain (Demographics, Job & Role, Compensation, Satisfaction, Tenure) with instant probability predictions and risk level badges.
3. **Model Performance:** Comprehensive confusion matrix heatmap, classification report, and metric cards.
4. **Feature Importance:** Top 15 key factors driving employee attrition risk shown via interactive Plotly bar charts.
5. **Dataset Overview:** Data explorer with summary statistics, column data types, and missing value checks.
6. **About Project:** Architecture details, deployment instructions, and local execution steps.

---

## 📁 Project Structure
```
employee-attrition-prediction/
│
├── app.py                            # Streamlit Multi-Page Web Application
├── train_model.py                    # ML Model Training & Pipeline Serialization
├── utils.py                          # UI Custom CSS & Plotly Visualizations
├── requirements.txt                  # Python Project Dependencies
├── employee_attrition_pipeline.pkl   # Serialized ML Pipeline (Joblib)
├── model_metrics.json                # Cached Performance & Feature Metrics
├── WA_Fn-UseC_-HR-Employee-Attrition.csv  # IBM HR Dataset
├── README.md                         # Project Documentation
└── .gitignore                        # Git Ignored Files
```

---

## 🚀 How to Run Locally

### 1. Clone or Open Workspace
```bash
cd "d:/employee attribute"
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model Pipeline
```bash
python train_model.py
```

### 4. Launch the Streamlit Web App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to use the application.

---

## ☁️ How to Deploy on Streamlit Community Cloud
1. Upload your code to a public GitHub repository.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub account and select your repository.
4. Set **Main file path** to `app.py`.
5. Click **Deploy!**

---

## 🔮 Future Improvements
- Integrate SHAP (SHapley Additive exPlanations) for individual prediction explainability.
- Add scenario simulation ("What-if" analysis for salary hikes or overtime reduction).
- Support batch CSV prediction for entire departments.
