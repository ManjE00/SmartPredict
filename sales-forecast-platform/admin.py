# admin.py
import streamlit as st
import pandas as pd
from sales_forecast.data_validation import validate_data
import logging
from datetime import datetime


# Configure basic logging (for demonstration purposes)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def admin_view():
    """Displays the admin interface."""
    st.title("🛡️ Admin Dashboard")

    st.header("⚙️ System Configuration (Future Implementation)")
    with st.expander("Data Integration Settings"):
        st.subheader("Configure External Data Sources")
        st.markdown("Here you can manage connections to external databases or APIs.")

        integration_type = st.selectbox(
            "Integration Type",
            ["None", "Database", "API"],
            help="Select the type of external data source to integrate with."
        )

        if integration_type == "Database":
            st.subheader("Database Settings")
            database_type = st.selectbox("Database Type", ["MySQL", "PostgreSQL", "Other"])
            database_host = st.text_input("Host")
            database_port = st.number_input("Port", value=3306)
            database_name = st.text_input("Database Name")
            database_user = st.text_input("User")
            database_password = st.text_input("Password", type="password")

            if st.button("Test Connection (Future)"):
                st.info("Testing database connection... (Future implementation)")
                logging.info(f"Admin attempted to test database connection: {database_type}, {database_host}")

            if st.button("Save Database Settings (Future)"):
                st.success("Database settings saved (Future implementation).")
                logging.info(f"Admin saved database settings: {database_type}, {database_host}")

        elif integration_type == "API":
            st.subheader("API Settings")
            api_url = st.text_input("API Endpoint URL")
            api_key = st.text_input("API Key", type="password")

            if st.button("Test Connection (Future)"):
                st.info("Testing API connection... (Future implementation)")
                logging.info(f"Admin attempted to test API connection: {api_url}")

            if st.button("Save API Settings (Future)"):
                st.success("API settings saved (Future implementation).")
                logging.info(f"Admin saved API settings: {api_url}")

        else:
            st.info("No data integration configured.")

    with st.expander("Thresholds and Alerts Configuration"):
        st.subheader("Define Alert Rules")
        st.markdown("Set up rules to trigger alerts based on sales performance or forecast deviations.")

        if 'alert_rules' not in st.session_state:
            st.session_state['alert_rules'] = []

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            alert_metric = st.selectbox(
                "Alert Metric",
                ["Sales", "Forecast Deviation (%)", "Customer Visits"],
                key="new_alert_metric",
                help="Select the metric to monitor for alerts."
            )
        with col2:
            threshold_type = st.selectbox(
                "Threshold Type",
                ["Below", "Above"],
                key="new_threshold_type",
                help=f"Trigger an alert when {alert_metric} is below or above this threshold."
            )
        with col3:
            threshold_value = st.number_input(
                f"Threshold Value for {alert_metric}",
                value=0.0 if alert_metric != "Forecast Deviation (%)" else 5.0,
                step=1.0 if alert_metric != "Forecast Deviation (%)" else 0.5,
                format="%.2f",
                key="new_threshold_value"
            )
        with col4:
            alert_level = st.selectbox(
                "Alert Level",
                ["Warning", "Critical"],
                key="new_alert_level",
                help="Severity level of the alert."
            )
        with col5:
            if st.button("Add Rule", key="add_alert_button"):
                new_rule = {
                    "metric": alert_metric,
                    "type": threshold_type,
                    "value": threshold_value,
                    "level": alert_level
                }
                st.session_state['alert_rules'].append(new_rule)
                st.success(f"New alert rule added: Trigger when **{alert_metric}** is **{threshold_type}** **{threshold_value}** (Level: **{alert_level}**)")
                logging.info(f"Admin added alert rule: {new_rule}")

        st.subheader("Current Alert Rules")
        if st.session_state['alert_rules']:
            for index, rule in enumerate(st.session_state['alert_rules']):
                col1_rule, col2_rule, col3_rule, col4_rule, col5_rule = st.columns([2, 1, 2, 1, 1])
                with col1_rule:
                    st.markdown(f"**Metric:** {rule['metric']}")
                    st.markdown(f"**Condition:** {rule['type']} {rule['value']}")
                with col2_rule:
                    st.markdown(f"**Level:** {rule['level']}")
                with col5_rule:
                    if st.button("Delete", key=f"delete_rule_{index}"):
                        del st.session_state['alert_rules'][index]
                        st.rerun() # Force a re-render to update the list
        else:
            st.info("No alert rules configured yet.")

    with st.expander("User Role Management"):
        st.subheader("Manage User Roles")
        st.markdown("Assign and modify roles for users of the platform.")

        if 'users' not in st.session_state:
            st.session_state['users'] = {"user1": "viewer", "user2": "editor", "admin": "administrator"}

        users_list = list(st.session_state['users'].keys())
        selected_user = st.selectbox("Select User", users_list)

        roles = ["administrator", "editor", "viewer"]
        current_role = st.selectbox(f"New Role for {selected_user}", roles, index=roles.index(st.session_state['users'].get(selected_user, "viewer")))

        if st.button("Assign Role"):
            st.session_state['users'][selected_user] = current_role
            st.success(f"Role '{current_role}' assigned to '{selected_user}'.")
            logging.info(f"Admin assigned role '{current_role}' to user '{selected_user}'.")

        st.subheader("Add New User")
        new_username = st.text_input("New Username", key="new_username")
        new_password = st.text_input("New Password", type="password", key="new_password")
        new_user_role = st.selectbox("Role for New User", roles, key="new_user_role")
        if st.button("Add User"):
            if new_username and new_username not in st.session_state['users']:
                st.session_state['users'][new_username] = new_user_role
                st.success(f"User '{new_username}' with role '{new_user_role}' added.")
                logging.info(f"Admin added user '{new_username}' with role '{new_user_role}'.")
            elif new_username in st.session_state['users']:
                st.error("Username already exists.")
            else:
                st.warning("Please enter a username.")

        st.subheader("Delete User")
        user_to_delete = st.selectbox("Select User to Delete", users_list, key="delete_user")
        if st.button("Delete User"):
            if user_to_delete != "admin": # Prevent accidental admin deletion (for this simple demo)
                del st.session_state['users'][user_to_delete]
                st.success(f"User '{user_to_delete}' deleted.")
                logging.warning(f"Admin deleted user '{user_to_delete}'.")
            else:
                st.error("Cannot delete the primary admin user in this demo.")
            st.rerun() # Update the user list

    st.header("✅ Data Validation")
    uploaded_file_validate = st.file_uploader("Upload CSV file for validation", type="csv", key="admin_validation_uploader")
    if uploaded_file_validate is not None:
        df_validate = pd.read_csv(uploaded_file_validate)
        st.subheader("Sample of Uploaded Data")
        st.dataframe(df_validate.head())
        if validate_data(df_validate):
            st.success("Data validation passed!")
            logging.info("Admin validated data successfully.")
        else:
            st.error("Data validation failed. Check for missing 'month' or 'sales' columns.")
            logging.warning("Admin data validation failed due to missing columns.")

        st.subheader("🔎 Enhanced Duplicate Detection")
        duplicate_columns = st.multiselect("Select columns to check for duplicates", df_validate.columns)
        if duplicate_columns:
            duplicates = df_validate[df_validate.duplicated(subset=duplicate_columns, keep=False)]
            if not duplicates.empty:
                st.warning(f"Found {len(duplicates)} duplicate rows based on selected columns:")
                st.dataframe(duplicates)

                # Option to download duplicate rows
                csv = duplicates.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Duplicate Rows as CSV",
                    data=csv,
                    file_name="duplicate_rows.csv",
                    mime="text/csv",
                )
                logging.warning(f"Admin found and downloaded {len(duplicates)} duplicate rows.")
            else:
                st.info("No duplicate rows found based on selected columns.")
                logging.info("Admin checked for duplicates - none found.")
        else:
            st.info("Please select columns to check for duplicates.")

    st.header("⚠️ Fault Detection and Impact Regulation (Future Implementation)")
    with st.expander("System Logs"):
        st.subheader("Recent System Logs")
        if 'system_logs' not in st.session_state:
            st.session_state['system_logs'] = []

        for log_entry in st.session_state['system_logs']:
            st.text(log_entry)

        if st.button("Clear Logs"):
            st.session_state['system_logs'] = []
            st.rerun()

        # Simulate logging some events when admin actions occur
        if st.session_state.get('alert_rules'):
            logging.info(f"Current alert rules: {st.session_state['alert_rules']}")
            st.session_state['system_logs'].append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - Displayed current alert rules.")

        if st.session_state.get('users'):
            logging.info(f"Current users and roles: {st.session_state['users']}")
            st.session_state['system_logs'].append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - Displayed current user roles.")

        if st.session_state.get('admin_validation_uploader') is not None:
            st.session_state['system_logs'].append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - Admin uploaded a file for validation.")

    with st.expander("Error Monitoring"):
        st.subheader("Recent Errors")
        if 'app_errors' not in st.session_state:
            st.session_state['app_errors'] = []

        for error_entry in st.session_state['app_errors']:
            st.error(error_entry)

        if st.button("Clear Errors"):
            st.session_state['app_errors'] = []
            st.rerun()

        # Simulate logging some errors (in a real app, these would be in try-except blocks)
        if not st.session_state.get('data_uploaded', False) and st.button("Simulate No Data Error"):
            error_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ERROR - User attempted to perform an action without uploading data."
            st.session_state['app_errors'].append(error_message)
            logging.error(error_message)
            st.rerun() # To show the error immediately

        if st.session_state.get('alert_rules') and not st.session_state['alert_rules'] and st.button("Simulate Alert Rule Error"):
            error_message = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - WARNING - No alert rules configured, but monitoring is active."
            st.session_state['app_errors'].append(error_message)
            logging.warning(error_message)
            st.rerun()

    with st.expander("Impact Regulation Tools"):
        st.subheader("Manual Intervention")
        st.markdown("Tools for manual intervention in case of critical issues.")
        st.warning("No specific tools implemented yet.")

        if st.button("Simulate Manual Intervention"):
            st.info("Manual intervention simulated. System state might have been adjusted (future functionality).")
            logging.warning("Admin triggered a simulated manual intervention.")

if __name__ == "__main__":
    st.set_page_config(page_title="Admin Dashboard", page_icon="🛡️", layout="wide")
    admin_view()