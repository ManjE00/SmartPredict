import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sales_forecast.model_training import train_random_forest_model, random_forest_forecast, gradient_boosting_forecast
from sales_forecast.visualization import create_sales_visualization
import logging

def forecast_section(processed_data, forecast_months, future_ads_spent, future_customer_visits, model_choice,
                     profit_margin, cost_per_ad, sales_target):
    st.header(f"Sales Forecast using {model_choice}")
    st.info(f"This forecast is generated using a {model_choice} model for the next {forecast_months} months.")

    if processed_data is not None and not processed_data.empty and 'month' in processed_data.columns:
        logging.debug("Data conditions met: processed_data is not None, not empty, and has 'month' column.")
        try:
            if model_choice == "Random Forest":
                st.info("Training Random Forest model...")
                trained_model, scaler, features, accuracy, actual_val, predicted_val, mae, mse, rmse, mape, smape = train_random_forest_model(
                    processed_data.copy())
                st.success("Random Forest model trained.")

                st.info("Preparing future data for Random Forest prediction...")
                last_date = pd.to_datetime(processed_data['month']).max()
                future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=forecast_months,
                                            freq='MS')
                future_df = pd.DataFrame({'month': future_dates,
                                          'ads_spent': [future_ads_spent] * forecast_months,
                                          'customer_visits': [future_customer_visits] * forecast_months})

                future_df['sales_lag_1'] = [processed_data['sales'].iloc[-1]] * forecast_months if not processed_data.empty else np.nan
                future_df['sales_lag_3'] = [processed_data['sales'].iloc[-3]] * forecast_months if len(
                    processed_data) >= 3 else np.nan
                future_df['sales_lag_1'] = future_df['sales_lag_1'].fillna(
                    processed_data['sales_lag_1'].mean() if 'sales_lag_1' in processed_data.columns else 0)
                future_df['sales_lag_3'] = future_df['sales_lag_3'].fillna(
                    processed_data['sales_lag_3'].mean() if 'sales_lag_3' in processed_data.columns else 0)

                st.info("Generating sales forecast using Random Forest...")
                forecast_predictions = random_forest_forecast(trained_model, scaler, features, future_df,
                                                                 processed_data)
                forecast_result = pd.DataFrame(
                    {'Month': future_dates, 'Forecasted Sales (RM)': forecast_predictions})
                st.success("Sales forecast generated using Random Forest.")

                st.subheader(f"Sales Forecast for the Next {forecast_months} Months (Random Forest)")

                if not forecast_result.empty:
                    last_historical_sales = processed_data['sales'].iloc[-1]
                    forecast_result['Percentage Change (%)'] = (
                                                                    (forecast_result['Forecasted Sales (RM)'] - last_historical_sales) / last_historical_sales) * 100
                    st.dataframe(forecast_result)

                    st.subheader(f"Accuracy of the {model_choice} Model")
                    st.write(f"R-squared (on test set): {accuracy:.2f}")
                    st.write(f"Mean Absolute Error (MAE): {mae:.2f} RM")
                    st.write(f"Mean Squared Error (MSE): {mse:.2f} RM^2")
                    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f} RM")
                    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                    st.write(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape:.2f}%")

                    # Create a DataFrame for actual vs. predicted
                    evaluation_df = pd.DataFrame({'Actual Sales': actual_val, 'Predicted Sales': predicted_val})
                    evaluation_df.index = actual_val.index  # Assuming index aligns with time

                    # Plot actual vs. predicted
                    fig_eval = px.line(evaluation_df, title="Actual vs. Predicted Sales (Test Set)")
                    st.plotly_chart(fig_eval, use_container_width=True)

                    with st.expander(f"Understanding the Accuracy (R-squared) for {model_choice}"):
                        st.info(f"""
                                            The accuracy score (R-squared) represents the proportion of the variance in the actual sales data that the {model_choice} model has captured.

                                            - A score closer to 100% (or 1.0) indicates that the model explains a large portion of the variability in past sales, and its predictions are likely to be closer to the actual values.
                                            - A score closer to 0% (or 0.0) suggests that the model does not explain much of the variability in past sales, and its predictions might not be very reliable.
                                            - A negative score indicates that the model performs worse than simply predicting the average of the historical sales.

                                            **A score of {accuracy:.2f}% is what we observed on the test data.** This gives an indication of how well the model might perform on unseen future data.

                                            It's recommended to consider the MAE, MSE, and RMSE as well, which give you an idea of the magnitude of the errors in Ringgit Malaysia.
                                            """)

                    st.subheader("📈 Sales Forecast Visualization")
                    st.plotly_chart(create_sales_visualization(processed_data.reset_index(), forecast_result),
                                     use_container_width=True)

                    st.subheader("Feature Importance (Random Forest)")
                    if 'features' in locals():  # Check if features exists
                        importances = trained_model.feature_importances_
                        feature_names = features
                        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
                        st.dataframe(feature_importance_df)
                    else:
                        st.info("Feature importance not available.")

                    # Focus Notification (Random Forest)
                    if forecast_result['Forecasted Sales (RM)'].mean() > last_historical_sales:
                        st.info(
                            "📈 The forecast suggests a potential increase in sales. Consider optimizing your current ads spend and customer engagement strategies to capitalize on this growth.")
                    else:
                        st.warning(
                            "📉 The forecast indicates a potential decrease in sales. Analyze your ads spent efficiency and explore strategies to attract more customer visits to improve sales.")

                    csv = forecast_result.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast Data (Random Forest)",
                        data=csv,
                        file_name="sales_forecast_rf.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No Random Forest forecast results to display.")

            elif model_choice == "Gradient Boosting":
                st.info("Training Gradient Boosting model...")
                forecast_result, trained_model, accuracy, actual_val, predicted_val, mae, mse, rmse, mape, smape, features = gradient_boosting_forecast(
                    processed_data.copy(),
                    forecast_months,
                    future_ads_spent,
                    future_customer_visits
                )
                st.success("Gradient Boosting model trained.")
                st.success("Sales forecast generated using Gradient Boosting.")

                st.subheader(f"Sales Forecast for the Next {forecast_months} Months (Gradient Boosting)")

                if not forecast_result.empty:
                    last_historical_sales = processed_data['sales'].iloc[-1]
                    forecast_result['Percentage Change (%)'] = (
                                                                    (forecast_result['Forecasted Sales (RM)'] - last_historical_sales) / last_historical_sales) * 100
                    st.dataframe(forecast_result)

                    st.subheader(f"Accuracy of the {model_choice} Model")
                    st.write(f"R-squared (on test set): {accuracy:.2f}")
                    st.write(f"Mean Absolute Error (MAE): {mae:.2f} RM")
                    st.write(f"Mean Squared Error (MSE): {mse:.2f} RM^2")
                    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f} RM")
                    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                    st.write(f"Symmetric Mean Absolute Percentage Error (SMAPE): {smape:.2f}%")

                    # Create a DataFrame for actual vs. predicted
                    evaluation_df = pd.DataFrame({'Actual Sales': actual_val, 'Predicted Sales': predicted_val})
                    evaluation_df.index = actual_val.index  # Assuming index aligns with time

                    # Plot actual vs. predicted
                    fig_eval = px.line(evaluation_df, title="Actual vs. Predicted Sales (Test Set)")
                    st.plotly_chart(fig_eval, use_container_width=True)

                    with st.expander(f"Understanding the Accuracy (R-squared) for {model_choice}"):
                        st.info(f"""
                                            The accuracy score (R-squared) represents the proportion of the variance in the actual sales data that the {model_choice} model has captured.

                                            - A score closer to 100% (or 1.0) indicates that the model explains a large portion of the variability in past sales, and its predictions are likely to be closer to the actual values.
                                            - A score closer to 0% (or 0.0) suggests that the model does not explain much of the variability in past sales, and its predictions might not be very reliable.
                                            - A negative score indicates that the model performs worse than simply predicting the average of the historical sales.

                                            **A score of {accuracy:.2f}% is what we observed on the test data.** This gives an indication of how well the model might perform on unseen future data.

                                            It's recommended to consider the MAE, MSE, and RMSE as well, which give you an idea of the magnitude of the errors in Ringgit Malaysia.
                                            """)

                    st.subheader("📈 Sales Forecast Visualization")
                    st.plotly_chart(create_sales_visualization(processed_data.reset_index(), forecast_result),
                                     use_container_width=True)

                    st.subheader("Feature Importance (Gradient Boosting)")
                    if hasattr(trained_model, 'feature_importances_') and features is not None:
                        importances = trained_model.feature_importances_
                        feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
                        st.dataframe(feature_importance_df)
                    else:
                        st.info("Feature importance not available for this model or features are missing.")

                    # Focus Notification (Gradient Boosting)
                    if forecast_result['Forecasted Sales (RM)'].mean() > last_historical_sales:
                        st.info(
                            "🚀 The forecast shows promising sales growth. Focus on scaling successful ads campaigns and enhancing customer experience to retain the increasing traffic.")
                    else:
                        st.warning(
                            "⚠️ The forecast projects a potential dip in sales. Evaluate the ROI of your current ads spend and consider initiatives to boost customer visits, such as targeted promotions or loyalty programs.")

                    csv = forecast_result.to_csv(index=False)
                    st.download_button(
                        label="Download Forecast Data (Gradient Boosting)",
                        data=csv,
                        file_name="sales_forecast_gb.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No Gradient Boosting forecast results to display.")

        except Exception as e:
            st.error(f"An error occurred during forecasting: {e}")
            logging.error(f"Error during forecasting: {e}", exc_info=True)  # Log the full error with traceback
    elif processed_data is not None and not processed_data.empty and 'month' not in processed_data.columns:
        st.error("Error: The 'month' column is missing from the processed data in the forecast page.")
        logging.error("Error: The 'month' column is missing from the processed data.")
    else:
        st.info("Please upload historical data on the main page to generate a forecast.")
        logging.debug("No data to display forecast.  Checking st.session_state:")
        logging.debug(f"st.session_state keys: {st.session_state.keys()}")
        if st.session_state.get('data_uploaded', False) and not st.session_state.get('cleaned_data'):
            st.rerun()

def forecast_configuration():
    """Display the forecast configuration in the sidebar."""
    st.sidebar.header("⚙️ Forecast Configuration")
    forecast_months = st.sidebar.slider(
        "Forecast Period (months)",
        1, 12, 3,
        help="Select the number of future months you want to generate a sales forecast for."
    )
    future_ads_spent = st.sidebar.number_input(
        "Future Ads Budget (RM)",
        min_value=0.0, step=100.0, value=1000.0,
        help="Enter the total budget you plan to spend on advertising for the entire forecast period."
    )
    future_customer_visits = st.sidebar.number_input(
        "Future Customer Visits",
        min_value=0, step=10, value=500,
        help="Enter the total number of customer visits you anticipate during the forecast period."
    )
    model_choice = st.sidebar.selectbox(
        "Choose Forecasting Model",
        ["Random Forest", "Gradient Boosting"],
        help="Select the machine learning model you want to use for generating the forecast. Both models have different strengths and can provide varying results."
    )
    profit_margin = st.sidebar.number_input("Profit Margin (%)", min_value=0.0, max_value=100.0, value=20.0)
    cost_per_ad = st.sidebar.number_input("Cost Per Ad", min_value=0.0, value=10.0)
    sales_target = st.sidebar.number_input("Sales Target", min_value=0.0, value=5000.0)

    return forecast_months, future_ads_spent, future_customer_visits, model_choice, profit_margin, cost_per_ad, sales_target

if 'cleaned_data' in st.session_state:
    logging.debug("cleaned_data is in st.session_state")
    logging.debug(f"Type of cleaned_data: {type(st.session_state['cleaned_data'])}")
    if isinstance(st.session_state['cleaned_data'], pd.DataFrame):
        logging.debug("cleaned_data is a DataFrame")
        if 'month' in st.session_state['cleaned_data'].columns:
            forecast_months, future_ads_spent, future_customer_visits, model_choice, profit_margin, cost_per_ad, sales_target = forecast_configuration()
            forecast_section(
                st.session_state['cleaned_data'].copy(),
                forecast_months,
                future_ads_spent,
                future_customer_visits,
                model_choice,
                profit_margin,
                cost_per_ad,
                sales_target
            )
        else:
            st.error("Error: The 'month' column is missing in the processed data in the forecast page.")
            logging.error("Error: The 'month' column is missing in the processed data in the forecast page.")
    else:
        st.error("Error: The data in st.session_state['cleaned_data'] is not a DataFrame. Please ensure you upload a valid CSV file on the home page.")
        logging.error(f"Error: The data in st.session_state['cleaned_data'] is not a DataFrame. Type: {type(st.session_state.get('cleaned_data'))}, Value: {st.session_state.get('cleaned_data')}")
        # Optionally, don't force a rerun immediately, let the user correct the upload.
        # st.rerun()
else:
    st.info("Please upload historical data on the main page to generate a forecast.")
    logging.info("No data uploaded yet, prompting user to upload.")
    if st.session_state.get('data_uploaded', False) and not st.session_state.get('cleaned_data'):
        st.rerun()