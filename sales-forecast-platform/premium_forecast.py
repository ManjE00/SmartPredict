import pandas as pd
import streamlit as st
import sys
import os
from admin import admin_view
import matplotlib.pyplot as plt
import io

# Get the directory of the current file (premium_forecast.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (one level up)
project_root = os.path.dirname(current_dir)

# Add the project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to sys.path: {project_root}")  # Add this line for debugging

# Now import your modules *after* modifying sys.path
from sales_forecast import auth
from sales_forecast.data_validation import validate_data, clean_data, create_features
from sales_forecast.visualization import create_sales_visualization
from sales_forecast.model_training import train_random_forest_model, gradient_boosting_forecast
from sales_forecast.utils import get_feature_names


def format_for_display(data):
    """Formats the data for display in Streamlit."""
    display_data = data.copy()
    if 'month' in display_data:
        display_data['month'] = pd.to_datetime(display_data['month']).dt.strftime('%Y-%m')
    return display_data


def user_view():
    if st.session_state['user_role'] == 'admin':
        st.warning("This page is intended for regular users only. Administrators have a separate dashboard.")
    else:
        st.title("🏠 SmartPredict")

    uploaded_file = st.file_uploader(
        "Upload your sales data (CSV)",
        type="csv",
        help="Please upload a CSV file containing historical sales data. The file should ideally have columns for 'month' (YYYY-MM format), 'sales' (numeric), 'ads_spent' (numeric), and 'customer_visits' (numeric).",
        accept_multiple_files=False, # Only allow one file
    )

    with st.expander("Need help with the data format?"):
        st.markdown(
            """
            **Expected CSV Format:**

            The CSV file should contain columns with the following information:

            - **month:** Dates representing the time period of the sales data. Ensure the format is consistent (e.g., YYYY-MM or YYYY-MM-DD).
            - **sales:** The total sales for the corresponding month.
            - **ads_spent:** The amount spent on advertising for that month.
            - **customer_visits:** The number of customer visits during that month.

            **Example Data:**

            ```csv
            month,sales,ads_spent,customer_visits
            2024-01,1000,150,200
            2024-02,1200,180,250
            2024-03,1150,160,220
            ...
            ```

            You can download a sample CSV file below to see the expected format.
            """
        )
        sample_data = pd.DataFrame(
            {'month': pd.to_datetime(['2025-01-01', '2025-02-01']), 'sales': [100, 120], 'ads_spent': [20, 25],
             'customer_visits': [50, 60]})
        csv = sample_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Sample CSV",
            data=csv,
            file_name="sample_sales_data.csv",
            mime="text/csv",
        )

    if uploaded_file is not None:
        try:
            with st.spinner("Validating data..."):
                historical_data = pd.read_csv(uploaded_file)
                is_valid = validate_data(historical_data.copy())
                if not is_valid:
                    st.error("Data validation failed...")
                    return  # Stop if validation fails
                cleaned_data = clean_data(historical_data.copy())
                original_data = historical_data.copy()

            # Check if cleaned_data is a DataFrame *before* storing it
            if isinstance(cleaned_data, pd.DataFrame):
                st.session_state['cleaned_data'] = cleaned_data
                st.session_state['original_data'] = original_data
                st.session_state['historical_data'] = cleaned_data
                st.session_state['data_uploaded'] = True
                st.success("Data uploaded and processed successfully!")

                st.subheader("🧹 Cleaned Data Preview")
                display_df = format_for_display(cleaned_data.copy())  # Use cleaned_data for display
                st.dataframe(display_df)

                st.subheader("Original Data Preview")
                st.dataframe(original_data.head())

                st.subheader("⚙️ Forecast Configuration")
                forecast_months = st.slider(
                    "Forecast Period (months)",
                    1, 12, 3,
                    help="Select the number of future months you want to generate a sales forecast for."
                )
                future_ads_spent = st.number_input(
                    "Future Ads Budget (RM)",
                    min_value=0.0, step=100.0, value=1000.0,
                    help="Enter the total budget you plan to spend on advertising for the entire forecast period."
                )
                future_customer_visits = st.number_input(
                    "Future Customer Visits",
                    min_value=0, step=10, value=500,
                    help="Enter the total number of customer visits you anticipate during the forecast period."
                )
                model_choice = st.selectbox(
                    "Choose Forecasting Model",
                    ["Random Forest (Placeholder)", "Gradient Boosting (Placeholder)"],
                    help="Select the machine learning model you want to use for generating the forecast. Both models have different strengths and can provide varying results."
                )

                st.session_state['forecast_months'] = forecast_months
                st.session_state['future_ads_spent'] = future_ads_spent
                st.session_state['future_customer_visits'] = future_customer_visits
                st.session_state['model_choice'] = model_choice

                st.info("After configuring the forecast parameters, click '🔮 Generate Forecast' to see the predictions.")

                if st.button("🔮 Generate Forecast"):
                    st.write("--- Homepage Debug Before Navigation ---")
                    st.write(f"Homepage Session State Keys: {st.session_state.keys()}")
                    if 'cleaned_data' in st.session_state:
                        st.write("Homepage: 'cleaned_data' exists in session state.")
                        st.write(f"Homepage: Type of 'cleaned_data': {type(st.session_state['cleaned_data'])}")
                        # Placeholder for actual forecasting logic
                        last_date = pd.to_datetime(st.session_state['cleaned_data']['month']).max()
                        future_dates = pd.date_range(start=last_date, periods=forecast_months + 1, freq='M')[1:]
                        forecast = pd.DataFrame({
                            'month': future_dates,
                            'forecasted_sales': [150] * forecast_months  # Placeholder forecast values
                        })
                        st.session_state['forecast_results'] = forecast
                        st.session_state['navigated_to_forecast'] = True
                        if hasattr(st, "switch_page"):
                            st.switch_page("pages/sales_forecast_page.py")
                        else:
                            st.warning("Please upgrade Streamlit...")
                    else:
                        st.write("Homepage: 'cleaned_data' DOES NOT exist in session state.")
                else:
                    st.error("Error: Cleaned data is not a DataFrame. Please check your data processing steps.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            pass
    else:
        st.session_state['data_uploaded'] = False
        st.session_state['navigated'] = False
        st.info("Please upload your sales data to proceed.")


def main():
    """Main application function"""
    st.session_state.setdefault('logged_in', False)
    st.session_state.setdefault('user_role', None)
    st.session_state.setdefault('cleaned_data', None)
    st.session_state.setdefault('historical_data', None)
    st.session_state.setdefault('forecast_results', None) # Initialize forecast results

    if not st.session_state.logged_in:
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.subheader("🔒 Login")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            login_button = st.button("Login")

            if login_button:
                if username == "admin" and password == "admin":
                    st.session_state['logged_in'] = True
                    st.session_state['user_role'] = 'admin'
                    st.rerun()
                elif username == "user" and password == "user":
                    st.session_state['logged_in'] = True
                    st.session_state['user_role'] = 'user'
                    st.rerun()
                else:
                    st.error("Invalid credentials")

            st.markdown("---")
            st.markdown("About Our Website:")
            st.markdown(    """
    **SmartPredict** is your intelligent platform for advanced sales forecasting.
    Leverage the power of data-driven insights to predict future sales trends,
    optimize your inventory, and make smarter business decisions.
    Upload your historical sales data and unlock accurate forecasts with our
    intuitive tools and powerful AI models.
    """) # Replace with your actual about text

        with col_right:
            st.image("images/smartpredict_logo.png", width=300) # Ensure correct path to your logo

    else:
        st.sidebar.title(f"👋 Logged in as: {st.session_state['user_role']}")
        if st.sidebar.button("🔒 Logout"):
            st.session_state['logged_in'] = False
            st.session_state['user_role'] = None
            st.session_state['cleaned_data'] = None
            st.session_state['historical_data'] = None
            st.session_state['forecast_results'] = None
            st.rerun()

        if st.session_state['user_role'] == 'user':
            user_view() # Regular users see the sales forecast functionalities
        elif st.session_state['user_role'] == 'admin':
            admin_view() # Admin users see the admin dashboard

if __name__ == "__main__":
    st.set_page_config(page_title="SmartPredict", page_icon="📈", layout="wide")
    st.session_state.setdefault('data_uploaded', False)
    st.session_state.setdefault('navigated', False)
    main()