import streamlit as st
import pandas as pd

def display_live_metrics(historical_data):
    """Displays live business metrics with time period selection and comparison."""
    st.header("Live Business Metrics")

    if historical_data is not None and not historical_data.empty:
        historical_data['month'] = pd.to_datetime(historical_data['month'])
        historical_data = historical_data.set_index('month')
        historical_data = historical_data.sort_index()

        time_period = st.selectbox("Select Time Period", ["Last 7 Days", "Last 30 Days", "Current Month", "Overall"], index=2)

        if time_period == "Last 7 Days":
            recent_data = historical_data.last("7D")
        elif time_period == "Last 30 Days":
            recent_data = historical_data.last("30D")
        elif time_period == "Current Month":
            now = pd.to_datetime('now')
            recent_data = historical_data[(historical_data.index.year == now.year) & (historical_data.index.month == now.month)]
        else:
            recent_data = historical_data

        if not recent_data.empty:
            last_sales = recent_data['sales'].iloc[-1]
            last_ads_spent = recent_data['ads_spent'].iloc[-1]
            last_customer_visits = recent_data['customer_visits'].iloc[-1]

            st.metric(label="Current Sales (RM)", value=f"RM {last_sales:,.2f}")
            st.metric(label="Current Ads Spent (RM)", value=f"RM {last_ads_spent:,.2f}")
            st.metric(label="Current Customer Visits", value=f"{last_customer_visits:,}")

            if time_period != "Overall" and len(recent_data) > 1:
                previous_sales = recent_data['sales'].iloc[-2]
                sales_change = ((last_sales - previous_sales) / previous_sales) * 100 if previous_sales != 0 else float('-inf')

                st.metric(label="Sales Change (vs Previous)", value=f"{sales_change:.2f}%", delta=f"{sales_change:.2f}%")
                if sales_change > 5:
                    st.success("Sales are showing a positive trend compared to the previous period.")
                elif sales_change < -5:
                    st.warning("Sales are showing a negative trend compared to the previous period.")
                else:
                    st.info("Sales are relatively stable compared to the previous period.")

                previous_ads_spent = recent_data['ads_spent'].iloc[-2]
                ads_change = ((last_ads_spent - previous_ads_spent) / previous_ads_spent) * 100 if previous_ads_spent != 0 else float('-inf')
                st.metric(label="Ads Spent Change (vs Previous)", value=f"{ads_change:.2f}%", delta=f"{ads_change:.2f}%")

                previous_visits = recent_data['customer_visits'].iloc[-2]
                visits_change = ((last_customer_visits - previous_visits) / previous_visits) * 100 if previous_visits != 0 else float('-inf')
                st.metric(label="Customer Visits Change (vs Previous)", value=f"{visits_change:.2f}%", delta=f"{visits_change:.2f}%")

            elif time_period != "Overall":
                st.info("Not enough data to compare to the previous period for the selected time frame.")

            st.subheader("Trends")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.line_chart(recent_data['sales'])
                st.caption("Sales Trend")
            with col2:
                st.line_chart(recent_data['ads_spent'])
                st.caption("Ads Spent Trend")
            with col3:
                st.line_chart(recent_data['customer_visits'])
                st.caption("Customer Visits Trend")

        else:
            st.info(f"No data available for the selected time period: {time_period}")

    else:
        st.info("Please upload sales data on the main page to view live metrics.")

if 'historical_data' in st.session_state:
    if isinstance(st.session_state['historical_data'], pd.DataFrame):
        display_live_metrics(st.session_state['historical_data'].copy())
    else:
        st.info("Please upload sales data on the main page first.")
else:
    st.info("Please upload sales data on the main page to view live metrics.")