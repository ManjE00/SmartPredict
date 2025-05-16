import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from datetime import datetime
import streamlit as st
import xgboost as xgb
from prophet import Prophet


def time_series_cv(data, n_splits=3):
    """
    Generates train and validation sets for time series cross-validation using TimeSeriesSplit.

    Args:
        data (pd.DataFrame): The time series data with a DatetimeIndex.
        n_splits (int): The number of splits (folds) to create.

    Yields:
        tuple: (train_index, val_index) for each split.
    """
    tss = TimeSeriesSplit(n_splits=n_splits)
    for train_index, val_index in tss.split(data):
        yield train_index, val_index



def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """Calculates the Symmetric Mean Absolute Percentage Error (SMAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100



def random_forest_forecast(model, scaler, features, future_data, historical_data):
    """Predicts future sales using a trained Random Forest model."""
    if len(historical_data) >= 1 and 'sales' in historical_data.columns:
        future_data['sales_lag_1'] = historical_data['sales'].iloc[-1]
    else:
        future_data['sales_lag_1'] = np.nan
    if len(historical_data) >= 3 and 'sales' in historical_data.columns:
        future_data['sales_lag_3'] = historical_data['sales'].iloc[-3]
    else:
        future_data['sales_lag_3'] = np.nan

    if 'sales_lag_1' in future_data.columns:  # Ensure column exists in future_data
        future_data['sales_lag_1'] = future_data['sales_lag_1'].fillna(
            historical_data['sales_lag_1'].mean() if 'sales_lag_1' in historical_data.columns else 0)
    else:
        future_data['sales_lag_1'] = 0

    if 'sales_lag_3' in future_data.columns:  # Ensure column exists in future_data
        future_data['sales_lag_3'] = future_data['sales_lag_3'].fillna(
            historical_data['sales_lag_3'].mean() if 'sales_lag_3' in historical_data.columns else 0)
    else:
        future_data['sales_lag_3'] = 0

    # Ensure engineered features are also in future_data
    if 'month' in future_data.columns:
        future_data['month_of_year'] = pd.to_datetime(future_data['month']).dt.month
        future_data['quarter_of_year'] = pd.to_datetime(future_data['month']).dt.quarter
        future_data['year'] = pd.to_datetime(future_data['month']).dt.year
        future_data['ads_customer_interaction'] = future_data['ads_spent'] * future_data['customer_visits']
    elif 'ds' in future_data.columns:  # For Prophet
        future_data['month_of_year'] = pd.to_datetime(future_data['ds']).dt.month
        future_data['quarter_of_year'] = pd.to_datetime(future_data['ds']).dt.quarter
        future_data['year'] = pd.to_datetime(future_data['ds']).dt.year
        future_data['ads_customer_interaction'] = future_data['ads_spent'] * future_data[
            'customer_visits']  # Assuming these columns might be present if Prophet is extended

    future_scaled = scaler.transform(future_data[features])
    predictions = model.predict(future_scaled)
    return predictions



def train_random_forest_model(data, cv_splits=3):
    """Trains a Random Forest Regressor model with hyperparameter tuning using time series CV."""
    data = data.copy()
    data['month'] = pd.to_datetime(data['month'])
    data = data.set_index('month').sort_index()

    # Feature Engineering (as before)
    data['sales_lag_1'] = data['sales'].shift(1)
    data['sales_lag_3'] = data['sales'].shift(3)
    data['month_of_year'] = data.index.month
    data['quarter_of_year'] = data.index.quarter
    data['year'] = data.index.year
    data['ads_customer_interaction'] = data['ads_spent'] * data['customer_visits']
    data = data.dropna()

    features = ['ads_spent', 'customer_visits', 'sales_lag_1', 'sales_lag_3',
                'month_of_year', 'quarter_of_year', 'year', 'ads_customer_interaction']
    target = 'sales'
    X = data[features]
    y = data[target]

    # Feature scaling *before* CV to prevent leakage within each fold
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Hyperparameter grid for Random Forest
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5]
    }

    # TimeSeriesSplit for cross-validation
    tss = TimeSeriesSplit(n_splits=cv_splits)
    grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=tss,
                                    scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_rf.fit(X_scaled, y)  # Fit on the *scaled* data

    best_rf_model = grid_search_rf.best_estimator_

    # Evaluate on a final hold-out set (the last fold of the CV)
    train_index, test_index = list(time_series_cv(data, n_splits=cv_splits))[-1]
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train_scaled = scaler.transform(X_train)  # Scale based on the *full* training data
    X_test_scaled = scaler.transform(X_test)

    y_pred_test = best_rf_model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred_test)
    smape = symmetric_mean_absolute_percentage_error(y_test, y_pred_test)


    return best_rf_model, scaler, features, r2, y_test, y_pred_test, mae, mse, rmse, mape, smape



def gradient_boosting_forecast(data, forecast_months, future_ads_spent, future_customer_visits, cv_splits=3):
    """Trains a Gradient Boosting model with hyperparameter tuning and makes forecasts,
    also returning evaluation metrics, using TimeSeriesSplit for CV."""
    data = data.copy()
    data['month'] = pd.to_datetime(data['month'])
    data = data.set_index('month').sort_index()

    # Feature Engineering
    data['sales_lag_1'] = data['sales'].shift(1)
    data['sales_lag_3'] = data['sales'].shift(3)
    data['month_of_year'] = data.index.month
    data['quarter_of_year'] = data.index.quarter
    data['year'] = data.index.year
    data['ads_customer_interaction'] = data['ads_spent'] * data['customer_visits']
    data = data.dropna()

    features = ['ads_spent', 'customer_visits', 'sales_lag_1', 'sales_lag_3',
                'month_of_year', 'quarter_of_year', 'year', 'ads_customer_interaction']
    target = 'sales'
    X = data[features]
    y = data[target]

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # TimeSeriesSplit for cross-validation
    tss = TimeSeriesSplit(n_splits=cv_splits)
    param_grid_gb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2]
    }
    grid_search_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb, cv=tss,
                                    scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_gb.fit(X_scaled, y)

    best_gb_model = grid_search_gb.best_estimator_

    # Evaluate on a final hold-out set
    train_index, test_index = list(time_series_cv(data, n_splits=cv_splits))[-1]
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_pred_test = best_gb_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred_test)
    smape = symmetric_mean_absolute_percentage_error(y_test, y_pred_test)

    # Prepare future dataframe for forecasting (as before)
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_months, freq='MS')
    future_df = pd.DataFrame({'month': future_dates, 'ads_spent': [future_ads_spent] * forecast_months,
                              'customer_visits': [future_customer_visits] * forecast_months})
    # ... (feature engineering for future_df as before) ...
    future_df['sales_lag_1'] = [data['sales'].iloc[-1]] * forecast_months
    future_df['sales_lag_3'] = [data['sales'].iloc[-3]] * forecast_months if len(data) >= 3 else [
                                                                                                np.nan] * forecast_months
    future_df['month_of_year'] = future_df['month'].dt.month
    future_df['quarter_of_year'] = future_df['month'].dt.quarter
    future_df['year'] = future_df['month'].dt.year
    future_df['ads_customer_interaction'] = future_df['ads_spent'] * future_df['customer_visits']
    future_df = future_df.fillna(method='ffill')

    future_scaled = scaler.transform(future_df[features])
    forecast_predictions = best_gb_model.predict(future_scaled)
    forecast_result = pd.DataFrame({'Month': future_dates, 'Forecasted Sales (RM)': forecast_predictions})

    return forecast_result, best_gb_model, r2, y_test, y_pred_test, mae, mse, rmse, mape, smape, features



def train_gradient_boosting_model(historical_data, cv_splits=3):
    """Trains a Gradient Boosting Regressor model and returns evaluation metrics, using TimeSeriesSplit."""
    historical_data = historical_data.copy()
    if 'month' not in historical_data.columns:
        raise KeyError("The 'month' column is missing in the historical data.")
    historical_data['month'] = pd.to_datetime(historical_data['month'], errors='coerce')
    historical_data = historical_data.set_index('month').sort_index()
    historical_data = historical_data.dropna(subset=['sales', 'ads_spent', 'customer_visits'])
    historical_data['sales_lag_1'] = historical_data['sales'].shift(1)
    historical_data['sales_lag_3'] = historical_data['sales'].shift(3)
    historical_data['month_of_year'] = historical_data.index.month
    historical_data['quarter_of_year'] = historical_data.index.quarter
    historical_data['year'] = historical_data.index.year
    historical_data['ads_customer_interaction'] = historical_data['ads_spent'] * historical_data['customer_visits']
    historical_data = historical_data.dropna()

    features = ['ads_spent', 'customer_visits', 'sales_lag_1', 'sales_lag_3', 'month_of_year', 'quarter_of_year', 'year',
                'ads_customer_interaction']
    target = 'sales'

    X = historical_data[features]
    y = historical_data[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tss = TimeSeriesSplit(n_splits=cv_splits)
    param_grid_gb_no_tune = {
        'n_estimators': [100],
        'learning_rate': [0.1],
        'max_depth': [3],
        'min_samples_split': [2],
        'min_samples_leaf': [1]
    }
    grid_search_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid_gb_no_tune, cv=tss,
                                    scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_gb.fit(X_scaled, y)
    best_gb_model = grid_search_gb.best_estimator_

    train_index, test_index = list(time_series_cv(historical_data, n_splits=cv_splits))[-1]  # Use historical_data
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_pred = best_gb_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_test, y_pred)

    # The forecast_result isn't generated here, so returning None for now
    return None, best_gb_model, r2, y_test, y_pred, mae, mse, rmse, mape, smape, features



def prophet_forecast(data, forecast_months):
    """Trains a Prophet model and makes future forecasts."""
    df = data.copy()
    df = df[['month', 'sales']].rename(columns={'month': 'ds', 'sales': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values(by='ds')
    df = df.dropna()

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=forecast_months, freq='MS')
    forecast = model.predict(future)

    forecast_result = forecast[['ds', 'yhat']].rename(columns={'ds': 'Month', 'yhat': 'Forecasted Sales (RM)'})
    return forecast_result, model
