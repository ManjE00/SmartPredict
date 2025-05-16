import streamlit as st
from datetime import datetime
import time

USER_CREDENTIALS = {
    "user": {"password": "user123", "role": "user"},
    "admin": {"password": "admin123", "role": "admin"},
    "manager": {"password": "manager123", "role": "admin"}
}

def check_session():
    """Initialize session state variables."""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'login_count' not in st.session_state:
        st.session_state.login_count = 0
    if 'last_login' not in st.session_state:
        st.session_state.last_login = None
    if 'login_error' not in st.session_state:  # For more robust error display
        st.session_state.login_error = None

def login():
    """Handle user login with a more aesthetic interface."""
    check_session()
    st.title("🔐 Login to Sales Forecast Pro")
    st.markdown("Welcome! Please enter your credentials to access the application.")

    # Use a form for better organization
    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submit_button = st.form_submit_button("Log In")

    if submit_button:
        if (username in USER_CREDENTIALS and
            password == USER_CREDENTIALS[username]["password"]):
            st.success("Login successful!")
            st.session_state.logged_in = True
            st.session_state.role = USER_CREDENTIALS[username]["role"]
            st.session_state.login_count += 1
            st.session_state.last_login = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.login_error = None  # Clear any previous error
            st.rerun()
        else:
            st.session_state.login_count = st.session_state.get('login_count', 0) + 1
            st.session_state.login_error = "Invalid username or password. Please try again."  # Store the error message
            if st.session_state.login_count >= 3:
                st.warning("Too many failed login attempts. Please try again later.")
                st.stop()

    if st.session_state.login_error:  # Display the error message
        st.error(st.session_state.login_error)

def logout():
    """Handle user logout."""
    check_session()
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.last_login = None
    st.info("Logged out successfully!")
    st.rerun()

def main():
    """Main function to run the app."""
    # Initialize session state
    check_session()

    # Use a sidebar for logout
    if st.session_state.logged_in:
        st.sidebar.button("Logout", on_click=logout)

    if not st.session_state.logged_in:
        login()  # show login
    else:
        st.title("Welcome to Sales Forecast Pro")
        st.write(f"Logged in as: {st.session_state.role}")
        st.write(f"Last Login: {st.session_state.last_login}")
        st.write(f"Login Count: {st.session_state.login_count}")

if __name__ == "__main__":
    main()
