import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

def apply_custom_css():
    st.markdown("""
        <style>
        /* Clean minimal styling */
        .simple-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            margin-bottom: 0.5rem;
        }

        .simple-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: #64748B;
            text-transform: uppercase;
        }

        .simple-val {
            font-size: 1.6rem;
            font-weight: 700;
            color: #0F172A;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_pipeline():
    model_filename = "employee_attrition_pipeline.pkl"
    if os.path.exists(model_filename):
        try:
            return joblib.load(model_filename)
        except Exception as e:
            st.error(f"Error loading pipeline: {e}")
            return None
    return None

@st.cache_data
def load_metrics():
    metrics_filename = "model_metrics.json"
    if os.path.exists(metrics_filename):
        with open(metrics_filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def render_metric_card(title, value):
    st.markdown(f"""
        <div class="simple-card">
            <div class="simple-title">{title}</div>
            <div class="simple-val">{value}</div>
        </div>
    """, unsafe_allow_html=True)
