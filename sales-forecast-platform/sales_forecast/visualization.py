import pandas as pd
import plotly.graph_objects as go


# Function to create the sales visualization using Plotly
def create_sales_visualization(historical_data, forecast_result):
    """Generate a cleaner sales forecasting line graph"""
    historical_data['month'] = pd.to_datetime(historical_data['month'])
    historical_data = historical_data.sort_values(by='month')

    forecast_result['Month'] = pd.to_datetime(forecast_result['Month'])
    forecast_result = forecast_result.sort_values(by='Month')

    fig = go.Figure()

    # Historical Sales
    fig.add_trace(
        go.Scatter(
            x=historical_data['month'],
            y=historical_data['sales'],
            name="Historical Sales",
            line=dict(color='#1f77b4', width=3),
            hovertemplate="<b>%{x|%b %Y}</b><br>Sales: RM %{y:,.0f}<extra></extra>"
        )
    )

    # Forecasted Sales
    fig.add_trace(
        go.Scatter(
            x=forecast_result['Month'],
            y=forecast_result['Forecasted Sales (RM)'],
            name="Forecast",
            line=dict(color='#ff7f0e', width=3, dash='dot'),
            hovertemplate="<b>%{x|%b %Y}</b><br>Forecast: RM %{y:,.0f}<extra></extra>"
        )
    )

    fig.update_layout(
        title="<b>Sales Forecast (RM)</b>",
        xaxis_title="Month",
        yaxis_title="Sales Amount (RM)",
        hovermode="x unified",
        height=500,
        template="plotly_dark"
    )

    return fig