import streamlit as st
import requests
import json
import time

# Alpha Vantage API key (directly inserted as requested, but consider st.secrets for security)
ALPHA_VANTAGE_API_KEY=st.secrets["ALPHA_VANTAGE_API_KEY"]

def get_stock_quote(symbol):
    """Fetches the real-time (or near real-time) quote for a given stock symbol."""
    base_url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'GLOBAL_QUOTE',
        'symbol': symbol,
        'apikey': ALPHA_VANTAGE_API_KEY
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if 'Global Quote' in data:
            return data['Global Quote']
        else:
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None
    except json.JSONDecodeError as e:
        st.error(f"Error decoding data for {symbol}: {e}")
        return None

def live_financial_dashboard():
    st.title("Live Financial Data")
    stock_symbols = st.sidebar.text_input("Enter stock symbols (comma-separated):", "AAPL,GOOGL")
    symbols_list = [s.strip().upper() for s in stock_symbols.split(',')]
    data_placeholder = st.empty()

    while True:
        live_data = {}
        for symbol in symbols_list:
            quote = get_stock_quote(symbol)
            if quote:
                live_data[symbol] = {
                    "symbol": quote['01. symbol'],
                    "price": quote['05. price'],
                    "volume": quote['06. volume'],
                    "timestamp": quote['07. latest trading day']
                }

        if live_data:
            data_string = "### Real-time Stock Quotes:\n"
            for symbol, details in live_data.items():
                data_string += f"- **{details['symbol']}**: Price = {details['price']}, Volume = {details['volume']} (as of {details['timestamp']})\n"
            data_placeholder.markdown(data_string)
        else:
            data_placeholder.info("No stock data to display.")

        time.sleep(60) # Update every 60 seconds (adjust as needed, respecting rate limits)

if __name__ == "__main__":
    live_financial_dashboard()