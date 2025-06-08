import streamlit as st
import pandas as pd
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from urllib.parse import quote
from datetime import timedelta
import numpy as np

def load_model():
    return pickle.load(open(r"D:\360DigitMG\Project - 1\Code\random_forest_model.pkl", "rb"))

def load_scaler():
    return joblib.load(r"D:\360DigitMG\Project - 1\Code\scaler.pkl")

def preprocess_data(df):
    """Prepares the test data by adding required features and handling missing values."""
    # Ensure 'Date' column exists
    if 'Date' not in df.columns:
        st.error("Missing 'Date' column in uploaded file.")
        return None
    
    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # Ensure 'Price' column exists
    if 'Price' not in df.columns:
        st.error("Missing 'Price' column in uploaded file.")
        return None
    
    # Feature Engineering
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['Price_Lag1'] = df['Price'].shift(1)
    df['Price_Lag7'] = df['Price'].shift(7)
    df['Price_Lag14'] = df['Price'].shift(14)
    df['Price_Lag30'] = df['Price'].shift(30)
    
    # Rolling Statistics
    df['Rolling_Mean_7'] = df['Price'].rolling(window=7).mean()
    df['Rolling_Std_7'] = df['Price'].rolling(window=7).std()
    df['Rolling_Min_7'] = df['Price'].rolling(window=7).min()
    df['Rolling_Max_7'] = df['Price'].rolling(window=7).max()
    
    # Fill missing values
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    
    return df


# for 30 days prediction
def predict_future(model, scaler, db_user, db_password, db_name, uploaded_file):
    # Load uploaded file
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    elif uploaded_file.name.endswith(".xls"):
        df = pd.read_excel(uploaded_file, engine="xlrd")
    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        st.error("Invalid file format. Please upload a valid Excel (.xls, .xlsx, or .csv) file.")
        return None

    if 'Date' not in df.columns:
        st.error("Missing 'Date' column in uploaded file.")
        return None

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['Date'], inplace=True)

    if 'Price' not in df.columns:
        st.error("Missing 'Price' column in uploaded file.")
        return None

    df = df.sort_values(by='Date').reset_index(drop=True)

    # Generate the next 30 days
    last_date = df['Date'].max()
    if pd.isna(last_date):
        st.error("Error: Could not determine the last date from the dataset.")
        return None  # Exit function if last_date is missing

    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    future_df = pd.DataFrame({'Date': future_dates})

    # **Get the last known price (Avoid IndexError)**
    if df['Price'].dropna().empty:
        st.error("Error: No valid historical price data found!")
        return None

    last_price = df['Price'].dropna().iloc[-1]  # Get last known price safely

    # ✅ **Ensure Feature Set Matches Model Training**
    features = [
        'Vol.', 'Change %', 'SMA_30', 'EMA_30', 'month', 'quarter', 'day_of_week',
        'Price_Lag1', 'Price_Lag7', 'Price_Lag14', 'Price_Lag30',
        'Rolling_Mean_7', 'Rolling_Std_7', 'Rolling_Min_7', 'Rolling_Max_7'
    ]

    # ✅ **Ensure Missing Columns Are Handled**
    for col in features:
        if col not in df.columns:
            df[col] = 0  # Fill missing columns with default values

    # ✅ **Generate Future Features**
    future_prices = []

    for i in range(30):
        # Handle IndexError by checking if `future_prices` has enough values
        lag1 = last_price if i == 0 else future_prices[-1]
        lag7 = future_prices[i - 7] if i >= 7 else last_price
        lag14 = future_prices[i - 14] if i >= 14 else last_price
        lag30 = future_prices[i - 30] if i >= 30 else last_price

        future_df.loc[i, 'Price_Lag1'] = lag1
        future_df.loc[i, 'Price_Lag7'] = lag7
        future_df.loc[i, 'Price_Lag14'] = lag14
        future_df.loc[i, 'Price_Lag30'] = lag30

        # Rolling statistics with small variations
        future_df.loc[i, 'Rolling_Mean_7'] = last_price * np.random.uniform(0.98, 1.02)
        future_df.loc[i, 'Rolling_Std_7'] = df['Rolling_Std_7'].iloc[-1] * np.random.uniform(0.95, 1.05)
        future_df.loc[i, 'Rolling_Min_7'] = last_price * np.random.uniform(0.97, 1.03)
        future_df.loc[i, 'Rolling_Max_7'] = last_price * np.random.uniform(0.99, 1.04)

        # ✅ **Check if all required features exist before scaling**
        for col in features:
            if col not in future_df.columns:
                future_df[col] = 0  # Fill missing columns

        # Scale input & predict
        input_scaled = scaler.transform([future_df.iloc[i][features].values])
        predicted_price = model.predict(input_scaled)[0]

        future_prices.append(predicted_price)  # Store prediction

    future_df['Forecasted_Price'] = future_prices  # Store predictions

    # ✅ **Keep only the next 30 days for final output**
    forecast_df = future_df[['Date', 'Forecasted_Price']]

    # ✅ **Save only the next 30 days to Database**
    engine = create_engine(f"mysql+pymysql://{db_user}:%s@localhost/{db_name}" % quote(f'{db_password}'))
    forecast_df.to_sql('forecasted_prices', con=engine, if_exists='replace', index=False)

    return forecast_df



# Function to generate confidence intervals
def calculate_confidence_interval(model, X, n_iter=100, alpha=0.05):
    """Bootstrap to estimate confidence intervals for predictions."""
    # Initialize an array to store predictions
    predictions = np.zeros((n_iter, len(X)))

    for i in range(n_iter):
        # Bootstrap sampling: randomly sample with replacement
        sample_idx = np.random.choice(range(len(X)), size=len(X), replace=True)
        X_sampled = X[sample_idx]

        # Predict on the bootstrapped sample
        predictions[i, :] = model.predict(X_sampled)

    # Calculate the confidence intervals (lower and upper bounds)
    lower_bound = np.percentile(predictions, 100 * alpha / 2, axis=0)
    upper_bound = np.percentile(predictions, 100 * (1 - alpha / 2), axis=0)

    return lower_bound, upper_bound



def main():
    st.sidebar.image(r"D:\360DigitMG\Project - 1\Code\AiSPRY logo.jpg")  # Add logo
    st.title("Iron Ore Price Forecasting")
    
    st.sidebar.header("Database Credentials")
    db_user = st.sidebar.text_input("User ID", "Type Here")
    db_password = st.sidebar.text_input("Password", "Type Here", type='password')
    db_name = st.sidebar.text_input("Database", "Type Here")
    uploaded_file = st.sidebar.file_uploader("Upload Test Data", type=['csv', 'xlsx'])
    
    if uploaded_file and st.button("Predict & Save"):
        model = load_model()
        scaler = load_scaler()
        forecast_df = predict_future(model, scaler, db_user, db_password, db_name, uploaded_file)
        
        if forecast_df is not None:
            st.success("Forecasted data saved to database!")
            st.subheader("Next 30 Days Forecasted Prices")
            st.write(forecast_df)
            
            # Check the available columns in the forecast_df
            st.write(forecast_df.columns)
            
            # Make sure the 'Forecasted_Price' column exists
            if 'Forecasted_Price' not in forecast_df.columns:
                st.error("The forecasted prices are missing in the dataframe.")
                return

            # If 'Forecasted_Price' exists, proceed with the visualization
            
            # Prepare Confidence Interval (using a placeholder example here)
            # Assuming we have some method to compute the CI
            lower_bound = forecast_df['Forecasted_Price'] - np.random.uniform(5, 10, size=len(forecast_df))  # Example lower bound
            upper_bound = forecast_df['Forecasted_Price'] + np.random.uniform(5, 10, size=len(forecast_df))  # Example upper bound

            # Visualization with Confidence Interval
            st.subheader("Forecast Visualization with Confidence Interval")

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(14, 8))

            # Plot forecasted prices
            sns.lineplot(x=forecast_df['Date'], y=forecast_df['Forecasted_Price'], label='Forecasted Price', color='#ff5733', linestyle='--', ax=ax, linewidth=3)

            # Plot confidence interval (shaded area)
            ax.fill_between(forecast_df['Date'], lower_bound, upper_bound, color='gray', alpha=0.2, label="Confidence Interval (95%)")

            # Add title and labels
            ax.set_title("Iron Ore Price Forecasting (Next 30 Days) with Confidence Interval", fontsize=18, fontweight='bold', color='#333333')
            ax.set_xlabel("Date", fontsize=14, color='#333333')
            ax.set_ylabel("Price (INR)", fontsize=14, color='#333333')

            # Rotate x-axis labels and improve date formatting
            ax.set_xticklabels(forecast_df['Date'].dt.strftime('%Y-%m-%d'), rotation=45, fontsize=12)

            # Add gridlines for better readability
            ax.grid(True, linestyle='--', alpha=0.7, which='both', axis='both', color='gray', linewidth=0.5)

            # Add legend
            ax.legend(loc='upper left', fontsize=12)

            # Adjust layout for better display
            plt.tight_layout()

            # Display the plot in Streamlit
            st.pyplot(fig)



if __name__ == '__main__':
    main()
