import os
import pandas as pd
import logging
from datetime import datetime
import streamlit as st

def log_error(error_msg, notify_admin=False):
    """Centralized error handling"""
    logging.basicConfig(filename='app_errors.log', level=logging.ERROR)
    logging.error(f"{datetime.now()}: {error_msg}")
    
    if notify_admin and st.session_state.role == "admin":
        st.error(f"ADMIN ALERT: {error_msg}")
        # Could add email/SMS notifications here

def get_feature_names():
    """Returns the feature names used in training"""
    return ['month_sin', 'month_cos', 'quarter', 'ads_spent',
           'customer_visits', 'sales_lag1', 'sales_lag3',
           'sales_rolling_mean3', 'is_holiday']

def save_to_log(dataframe, model_name, mape_scores):
    """Save forecast log for future reference"""
    os.makedirs("logs", exist_ok=True)
    filename = f"logs/forecast_log_{pd.Timestamp.now()}.csv"
    dataframe.to_csv(filename, index=False)

def generate_report(data, report_type="pdf"):
    """Generate comprehensive business reports"""
    if report_type == "pdf":
        # Use libraries like ReportLab or PyPDF2
        pass
    elif report_type == "excel":
        writer = pd.ExcelWriter("business_report.xlsx")
        data.to_excel(writer, sheet_name="Sales Forecast")
        # Add more sheets
        writer.close()
    if report_type == "pdf":
        report_file = "business_report.pdf"
    elif report_type == "excel":
        report_file = "business_report.xlsx"
    return report_file

# sales_forecast/utils.py

def get_feature_names():
    """Return the list of feature names used for the forecast model."""
    return ['month_sin', 'month_cos', 'quarter', 'ads_spent', 'customer_visits',
            'sales_lag1', 'sales_lag3', 'sales_rolling_mean3', 'is_holiday']
