import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.stats.mstats import winsorize
from sqlalchemy import create_engine

# from hyperparameter_tuning.tune_svr import tune_svr
# from hyperparameter_tuning.tune_xgboost import tune_xgboost

# from models.arima_model import train_arima
# from models.sarima_model import train_sarima
# from models.lstm_model import train_lstm
# from models.gru_model import train_gru
from models.random_forest_model import train_random_forest
# from models.train_xgboost import train_xgboost
# from models.svr_model import train_svr
# from models.transformer_model import train_transformer

# from evaluation import compare_models

# Reading the file from local directory
data = pd.read_csv('Data_set_Iron.csv')

# Credentials to connect to Database
user = 'root'
pw = 'root'  # Use the password directly if there are no special characters
db = 'iron_ore_forecasting'  
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Uploading data to the database
data.to_sql('iron_ore_price', con=engine, if_exists='replace', chunksize=1000, index=False)

# Querying the database
sql = 'select * from iron_ore_price;'  # Reading the data from DB
df = pd.read_sql_query(sql, engine.connect())

# Number of rows and columns
print(df.shape)

# Data types and non-null counts
print(df.info())  

# First few rows
print(df.head())

# Strip column names to avoid accidental spaces
df.columns = df.columns.str.strip()

# Convert 'Date' column to datetime format and sort
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values(by='Date')

# Remove duplicate columns (Open, High, Low) since they are identical to Price
df = df.drop(columns=['Open', 'High', 'Low'])

# Convert 'Change %' to numeric
df['Change %'] = df['Change %'].str.replace('%', '').astype(float) / 100

# Convert 'Vol.' column to numeric after removing 'K'
df["Vol."] = df["Vol."].astype(str).str.replace("K", "", regex=True)
df["Vol."] = pd.to_numeric(df["Vol."], errors="coerce") * 1000

# **🔹 Missing Value Check and Handling**
print("\nMissing Values Before Handling:")
print(df.isnull().sum())

# Filling missing values
df["Vol."].fillna(df["Vol."].median(), inplace=True)  # Fill 'Vol.' with median
df["Price"].fillna(method="bfill", inplace=True)  # Fill 'Price' with backward fill
df["Change %"].fillna(0, inplace=True)  # Replace NaN in 'Change %' with 0

# **🔹 Missing Values After Handling**
print("\nMissing Values After Handling:")
print(df.isnull().sum())

# **Statistical Analysis**
print(df.describe())

# **First, Second, Third, and Fourth Moment Analysis**
print(f"Mean Price: {df['Price'].mean()}")
print(f"Variance: {df['Price'].var()}, Standard Deviation: {df['Price'].std()}")
print(f"Skewness: {skew(df['Price'])}")
print(f"Kurtosis: {kurtosis(df['Price'])}")

# **Data Visualization**
plt.figure(figsize=(8, 5))
sns.histplot(df['Price'], bins=30, kde=True)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# **Time Series Analysis**
plt.figure(figsize=(12, 6))
sns.lineplot(x=df['Date'], y=df['Price'], label='Price')
plt.title('Iron Ore Price Trend')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# **Moving Averages**
df['SMA_30'] = df['Price'].rolling(window=30).mean()
df['EMA_30'] = df['Price'].ewm(span=30, adjust=False).mean()

plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Price'], label='Market Price', alpha=0.5)
plt.plot(df['Date'], df['SMA_30'], label='30-Day SMA', linestyle='dashed')
plt.plot(df['Date'], df['EMA_30'], label='30-Day EMA', linestyle='dotted')
plt.title('Market Price Trend with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Market Price')
plt.legend()
plt.show()

# **Outlier Handling using Winsorization**
df['Price'] = winsorize(df['Price'], limits=[0.05, 0.05])

plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Price'])
plt.title('Box Plot After Winsorization')
plt.show()

# **Feature Engineering**
df['month'] = df['Date'].dt.month
df['quarter'] = df['Date'].dt.quarter
df['day_of_week'] = df['Date'].dt.dayofweek

# **🔹 Missing Value Check for Feature Engineering**
print("\nMissing Values After Feature Engineering:")
print(df.isnull().sum())

# Fill missing values if any remain
df.fillna(method="bfill", inplace=True)

# **Hyperparameter Tuning for XGBoost**
# print("\nTuning XGBoost...")
# xgboost_best_params, xgb_rmse, xgb_mae, xgb_mape = tune_xgboost(df)
# print(f"XGBoost Tuned - RMSE: {xgb_rmse:.4f}, MAE: {xgb_mae:.4f}, MAPE: {xgb_mape:.2f}%")

# # **Hyperparameter Tuning for SVR**
# print("\nTuning SVR...")
# svr_best_params, svr_rmse, svr_mae, svr_mape = tune_svr(df, n_trials=3, test_size=0.1)
# print(f"SVR Tuned - RMSE: {svr_rmse:.4f}, MAE: {svr_mae:.4f}, MAPE: {svr_mape:.2f}%")

# **Export Processed Data to CSV**
df.to_csv('processed_iron_ore_data.csv', index=False)
print("Processed data exported successfully to 'processed_iron_ore_data.csv'")


# evaluating different models
results = {}

# Train Models with Tuned Parameters
# results["XGBoost"] = train_xgboost(df, xgboost_best_params)
# results["SVR"] = train_svr(df, svr_best_params)

# # Train & Evaluate ARIMA
# results["ARIMA"] = train_arima(df)

# # Train & Evaluate SARIMA
# results["SARIMA"] = train_sarima(df)

# # Train & Evaluate LSTM
# results["LSTM"] = train_lstm(df)

# # Train & Evaluate GRU
# results["GRU"] = train_gru(df)

# # Train & Evaluate Random Forest
results["Random Forest"] = train_random_forest(df)

# # Access values safely
# print(results["Random Forest"]["rmse"])
# print(results["Random Forest"]["model_path"])

# #Train & Evaluate TRANSFORMER
# results["Transformer"] = train_transformer(df)

# # Compare all models
# compare_models(results)


# # **🔹 Missing Value Handling for Future Data**
# latest_values = df[features].iloc[-1]
# for col in features:
#     future_df[col] = latest_values[col]

# # Apply the same scaling transformation
# future_X_scaled = pd.DataFrame(scaler.transform(future_df[features]), columns=features)

# # Make predictions
# future_predictions = model.predict(future_X_scaled)

# # **Plot Future Predictions**
# plt.figure(figsize=(12, 6))
# plt.plot(future_dates, future_predictions, label='Predicted Market Price', linestyle='dashed', color='red')
# plt.title('Future Market Price Predictions')
# plt.xlabel('Date')
# plt.ylabel('Predicted Price')
# plt.legend()
# plt.show()
