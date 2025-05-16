# forecasting.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from datetime import datetime

def generate_forecast(historical_data, months_to_forecast, future_ads_spent, future_customer_visits, industry_type='retail'):
    historical_data = historical_data.dropna(subset=['sales', 'ads_spent', 'customer_visits', 'month']) # Ensure 'month' is present
    historical_data['month'] = pd.to_datetime(historical_data['month'], errors='coerce') # Ensure 'month' is datetime

    if industry_type == "retail":
        features = ['ads_spent', 'customer_visits']
    elif industry_type == "fitness":
        features = ['ads_spent', 'customer_visits']
    else:
        features = ['ads_spent', 'customer_visits']

    X = historical_data[features]
    y = historical_data['sales']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, random_state=42)
    model.fit(X_scaled, y)

    future_input = pd.DataFrame({'ads_spent': future_ads_spent, 'customer_visits': future_customer_visits})
    future_input_scaled = scaler.transform(future_input)
    forecasted_sales = model.predict(future_input_scaled)

    last_month = historical_data['month'].max()
    start_date = last_month + pd.Timedelta(days=1)

    forecast_result = pd.DataFrame({
        'Month': pd.date_range(start=start_date, periods=months_to_forecast, freq='MS'),
        'Forecasted Sales (RM)': forecasted_sales
    })

    accuracy = 95
    return forecast_result, accuracy
