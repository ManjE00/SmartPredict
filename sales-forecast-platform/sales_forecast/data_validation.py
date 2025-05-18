# sales_forecast/data_validation.py
import pandas as pd
import streamlit as st
from datetime import datetime
from scipy import stats
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def validate_data(data):
    required_columns = ['month', 'sales', 'ads_spent', 'customer_visits']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"Error: Missing required columns: {', '.join(missing_columns)}. Please ensure your CSV contains 'month', 'sales', 'ads_spent', and 'customer_visits'.")
        return False
    missing_values = data[required_columns].isnull().any()
    if missing_values.any():
        missing_cols_with_na = missing_values[missing_values].index.tolist()
        st.error(f"Error: Missing values found in the following required columns: {', '.join(missing_cols_with_na)}. Please ensure there are no empty cells in 'month', 'sales', 'ads_spent', and 'customer_visits'.")
        return False
    try:
        pd.to_datetime(data['month'], errors='raise')
    except ValueError:
        st.error("Error: Incorrect format in the 'month' column. Please ensure it is in a recognizable date format (e.g., %Y-%m-%d, MM/DD/YYYY).")
        return False
    return True

def clean_data(data):
    cleaned_data = data.copy()
    print("Inside clean_data: Initial type of cleaned_data:", type(cleaned_data)) # DEBUG

    initial_rows = len(cleaned_data)
    cleaned_data = cleaned_data.dropna(subset=['sales', 'ads_spent', 'customer_visits', 'month'])
    print("Inside clean_data: Type after dropna:", type(cleaned_data)) # DEBUG
    print("Inside clean_data: Shape after dropna:", cleaned_data.shape) # DEBUG

    rows_after_dropna = len(cleaned_data)
    if initial_rows - rows_after_dropna > 0:
        st.warning(
            f"Warning: Removed {initial_rows - rows_after_dropna} rows with missing values in 'month', 'sales', 'ads_spent', or 'customer_visits'.")

    cleaned_data['month'] = pd.to_datetime(cleaned_data['month'], errors='coerce')
    print("Inside clean_data: Type after to_datetime:", type(cleaned_data)) # DEBUG

    cleaned_data = cleaned_data.sort_values(by='month')
    print("Inside clean_data: Type after sort_values:", type(cleaned_data)) # DEBUG

    duplicate_rows = cleaned_data[cleaned_data.duplicated(subset=['month'], keep='first')]
    cleaned_data = cleaned_data.drop_duplicates(subset=['month'], keep='first')
    print("Inside clean_data: Type after drop_duplicates:", type(cleaned_data)) # DEBUG
    print("Inside clean_data: Shape after drop_duplicates:", cleaned_data.shape) # DEBUG

    if not duplicate_rows.empty:
        st.warning(
            f"Warning: Removed {len(duplicate_rows)} duplicate entries based on the 'month' column, keeping the first occurrence for each month.")

    cleaned_data = cleaned_data.reset_index(drop=True)
    print("Inside clean_data: Type after reset_index:", type(cleaned_data)) # DEBUG

    #  ISOLATE THE PROBLEM (Step 4)
    # cleaned_data_with_features = create_features(cleaned_data.copy())  # COMMENT OUT THIS LINE
    # print("Inside clean_data: Type after create_features:", type(cleaned_data_with_features)) # DEBUG
    # print("Inside clean_data: Shape after create_features:", cleaned_data_with_features.shape) # DEBUG

    # print("Inside clean_data: Type before return:", type(cleaned_data_with_features)) # DEBUG
    # return cleaned_data_with_features
    return cleaned_data # Just return the cleaned data WITHOUT features

def detect_outliers(data, threshold=3):
    z_scores = np.abs(stats.zscore(data[['sales', 'ads_spent', 'customer_visits']]))
    return (z_scores > threshold).any(axis=1)

def create_features(data):
    data = data.copy()
    print("Inside create_features: Initial type of data:", type(data)) # DEBUG

    data['month'] = pd.to_datetime(data['month'], errors='coerce')
    print("Inside create_features: Type after to_datetime:", type(data)) # DEBUG

    data = data.sort_values(by='month') # Sort the data by month
    data['forecast_month'] = data['month'].copy()
    data['year'] = data['month'].dt.year
    data['month_num'] = data['month'].dt.month
    data['day'] = data['month'].dt.day
    data['week'] = data['month'].dt.isocalendar().week.astype(int)
    data['quarter'] = data['month'].dt.quarter
    data['dayofweek'] = data['month'].dt.dayofweek
    data['dayofyear'] = data['month'].dt.dayofyear
    data['is_month_start'] = data['month'].dt.is_month_start.astype(int)
    data['is_month_end'] = data['month'].dt.is_month_end.astype(int)
    data['is_quarter_start'] = data['month'].dt.is_quarter_start.astype(int)
    data['is_quarter_end'] = data['month'].dt.is_quarter_end.astype(int)
    data['is_year_start'] = data['month'].dt.is_year_start.astype(int)
    data['is_year_end'] = data['month'].dt.is_year_end.astype(int)
    data['days_in_month'] = data['month'].dt.daysinmonth

    for lag in [1, 2, 3, 6, 12]:
        data[f'sales_lag_{lag}'] = data['sales'].shift(lag)

    for window in [3, 6]:
        data[f'sales_rolling_mean_{window}'] = data['sales'].rolling(window=window, min_periods=1).mean().shift(1)
        data[f'sales_rolling_std_{window}'] = data['sales'].rolling(window=window, min_periods=1).std().shift(1)

    data['month_sin'] = np.sin(2 * np.pi * data['month_num'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month_num'] / 12)

    data = data.drop(columns=['month']) # Drop the original datetime 'month' column
    data = data.rename(columns={'forecast_month': 'month'}) # Rename the copy back to 'month'
    data = data.dropna() # Drop rows with NaN values created by lagging and rolling
    print("Inside create_features: Type after dropna:", type(data)) # DEBUG
    print("Inside create_features: Shape after dropna:", data.shape) # DEBUG

    data = data.reset_index(drop=True) # Reset index after dropping NaNs
    print("Inside create_features: Type before return:", type(data)) # DEBUG
    return data