USE sales_forecasting;

-- Create `users` table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    role ENUM('user', 'admin') NOT NULL
);

-- Create `uploaded_data` table
CREATE TABLE IF NOT EXISTS uploaded_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    month DATE,
    sales FLOAT,
    is_holiday INT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Create `forecasts` table
CREATE TABLE IF NOT EXISTS forecasts (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    forecast_month DATE,
    forecasted_sales FLOAT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Create `settings` table (optional)
CREATE TABLE IF NOT EXISTS settings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    forecast_range INT DEFAULT 12,   -- Max forecast period in months
    data_retention INT DEFAULT 90,   -- Number of days for data retention
    accuracy_threshold INT DEFAULT 85 -- Minimum accuracy threshold for forecast
);

SHOW TABLES;
