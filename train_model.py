import os
import json
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

def train_and_evaluate():
    data_path = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file {data_path} not found!")

    print("--- STEP 1: DATA LOADING & INSPECTION ---")
    df = pd.read_csv(data_path)
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    missing_vals = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    print(f"Missing Values: {missing_vals}")
    print(f"Duplicate Records: {duplicate_rows}")
    
    attrition_counts = df['Attrition'].value_counts().to_dict()
    print(f"Attrition Distribution: {attrition_counts}")

    print("\n--- STEP 2: DATA PREPROCESSING ---")
    # Clean duplicates if any
    if duplicate_rows > 0:
        df = df.drop_duplicates()

    # Drop constant/non-informative technical columns
    cols_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    df_clean = df.drop(columns=existing_drops)

    # Separate target
    X = df_clean.drop(columns=['Attrition'])
    y = df_clean['Attrition'].map({'Yes': 1, 'No': 0})

    # Define categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"Categorical Features ({len(categorical_cols)}): {categorical_cols}")
    print(f"Numerical Features ({len(numerical_cols)}): {numerical_cols}")

    # Build ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )

    # Full ML Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])

    print("\n--- STEP 3: MODEL TRAINING ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train Set: {X_train.shape[0]} samples | Test Set: {X_test.shape[0]} samples")

    pipeline.fit(X_train, y_train)

    print("\n--- STEP 4: MODEL EVALUATION ---")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec_yes = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_yes = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_yes = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

    prec_no = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
    rec_no = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
    f1_no = f1_score(y_test, y_pred, pos_label=0, zero_division=0)

    cm = confusion_matrix(y_test, y_pred).tolist()
    report_dict = classification_report(y_test, y_pred, target_names=['No', 'Yes'], output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=['No', 'Yes'])

    print(f"Accuracy: {acc:.4f}")
    print(f"Attrition = Yes | Precision: {prec_yes:.4f}, Recall: {rec_yes:.4f}, F1: {f1_yes:.4f}")
    print(f"Attrition = No  | Precision: {prec_no:.4f}, Recall: {rec_no:.4f}, F1: {f1_no:.4f}")
    print("\nConfusion Matrix:")
    print(np.array(cm))
    print("\nClassification Report:\n", report_str)

    print("\n--- STEP 5: FEATURE IMPORTANCE ---")
    rf_model = pipeline.named_steps['classifier']
    prep = pipeline.named_steps['preprocessor']
    feature_names_out = prep.get_feature_names_out()
    
    # Clean feature names for UI presentation
    cleaned_feature_names = []
    for f in feature_names_out:
        f_clean = f.replace('num__', '').replace('cat__', '')
        cleaned_feature_names.append(f_clean)

    importances = rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'feature': cleaned_feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    top_features = feat_imp_df.head(20).to_dict(orient='records')
    print("Top 10 Features:")
    for row in top_features[:10]:
        print(f"  {row['feature']}: {row['importance']:.4f}")

    print("\n--- STEP 6: SAVE MODEL PERSISTENCE & METRICS ---")
    model_filename = "employee_attrition_pipeline.pkl"
    joblib.dump(pipeline, model_filename)
    print(f"Saved pipeline to {model_filename}")

    # Save metrics JSON for Streamlit interface
    metrics_data = {
        "dataset_info": {
            "total_records": int(df.shape[0]),
            "total_features": int(X.shape[1]),
            "attrition_yes_count": int(attrition_counts.get('Yes', 0)),
            "attrition_no_count": int(attrition_counts.get('No', 0)),
            "missing_values": int(missing_vals),
            "duplicates": int(duplicate_rows)
        },
        "performance": {
            "accuracy": float(acc),
            "precision_yes": float(prec_yes),
            "recall_yes": float(rec_yes),
            "f1_yes": float(f1_yes),
            "precision_no": float(prec_no),
            "recall_no": float(rec_no),
            "f1_no": float(f1_no),
            "confusion_matrix": cm,
            "classification_report_dict": report_dict,
            "classification_report_str": report_str
        },
        "top_features": top_features,
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols
    }

    with open("model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
    print("Saved metrics to model_metrics.json")
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    train_and_evaluate()
