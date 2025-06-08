import numpy as np
import pandas as pd
import pickle
import joblib
import optuna
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from evaluation import evaluate_model

def preprocess_data(df):
    """Feature Engineering: Adds lag features, rolling statistics & fills missing values."""
    df = df.copy()
    
    # **Creating Lag Features**
    df['Price_Lag1'] = df['Price'].shift(1)
    df['Price_Lag7'] = df['Price'].shift(7)
    df['Price_Lag14'] = df['Price'].shift(14)
    df['Price_Lag30'] = df['Price'].shift(30)

    # **Rolling Features**
    df['Rolling_Mean_7'] = df['Price'].rolling(window=7, min_periods=1).mean()
    df['Rolling_Std_7'] = df['Price'].rolling(window=7, min_periods=1).std()
    df['Rolling_Min_7'] = df['Price'].rolling(window=7, min_periods=1).min()
    df['Rolling_Max_7'] = df['Price'].rolling(window=7, min_periods=1).max()

    # Fill NaN values
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    return df

def objective(trial, X_train, X_test, y_train, y_test):
    """Objective function for Optuna hyperparameter tuning"""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, None])  # ✅ Fix: 'None' as string
    }

    # ✅ Convert 'None' back to actual None in RandomForestRegressor
    max_features = None if params['max_features'] == 'None' else params['max_features']

    # ✅ Create model with correct max_features
    model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        max_features=max_features,  # Use 'sqrt', 'log2' or an integer
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse, mae, mape = evaluate_model(y_test, y_pred, "Random Forest")
    
    return rmse  # Minimize RMSE


def train_random_forest(df, n_trials=10):
    """Train Random Forest with hyperparameter tuning and save the model."""
    df = preprocess_data(df)

    features = ['Vol.', 'Change %', 'SMA_30', 'EMA_30', 'month', 'quarter', 'day_of_week',
                'Price_Lag1', 'Price_Lag7', 'Price_Lag14', 'Price_Lag30',
                'Rolling_Mean_7', 'Rolling_Std_7', 'Rolling_Min_7', 'Rolling_Max_7']
    
    X = df[features]
    y = df['Price']

    # **Scaling**
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # **Train-test split**
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=False)

    # **Hyperparameter Tuning with Optuna**
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=n_trials)

    # **Train Best Model**
    best_params = study.best_params
    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)

    # **Predictions & Evaluation**
    y_pred = model.predict(X_test)
    rmse, mae, mape = evaluate_model(y_test, y_pred, "Random Forest")

    # **Save Model & Scaler**
    with open("random_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)
    joblib.dump(scaler, "scaler.pkl")

    # **Future Predictions (30 Days)**
    future_dates = pd.date_range(start=df['Date'].max(), periods=30, freq='D')
    future_df = pd.DataFrame({'Date': future_dates})

    # **Generate synthetic future values**
    future_df['month'] = future_df['Date'].dt.month
    future_df['quarter'] = future_df['Date'].dt.quarter
    future_df['day_of_week'] = future_df['Date'].dt.dayofweek
    future_df['SMA_30'] = df['SMA_30'].iloc[-1] * (1 + np.random.uniform(-0.02, 0.02, 30))
    future_df['EMA_30'] = df['EMA_30'].iloc[-1] * (1 + np.random.uniform(-0.015, 0.015, 30))
    future_df['Vol.'] = df['Vol.'].iloc[-1] * (1 + np.random.uniform(-0.03, 0.03, 30))
    future_df['Change %'] = np.random.uniform(-0.02, 0.02, 30)

    # Ensure rolling features exist in future predictions
    for col in ['Rolling_Mean_7', 'Rolling_Std_7', 'Rolling_Min_7', 'Rolling_Max_7']:
        if col not in future_df.columns:
            future_df[col] = future_df['SMA_30']

    # **Iterative Forecasting**
    last_price = df['Price'].iloc[-1]  # Get last known price
    future_prices = []

    for i in range(30):
        lag1 = last_price if i == 0 else future_prices[-1]  # Use previous day's prediction
        lag7 = future_prices[i - 7] if i >= 7 else last_price
        lag14 = future_prices[i - 14] if i >= 14 else last_price
        lag30 = future_prices[i - 30] if i >= 30 else last_price

        future_df.loc[i, 'Price_Lag1'] = lag1
        future_df.loc[i, 'Price_Lag7'] = lag7
        future_df.loc[i, 'Price_Lag14'] = lag14
        future_df.loc[i, 'Price_Lag30'] = lag30

        input_scaled = scaler.transform([future_df.iloc[i][features].values])
        predicted_price = model.predict(input_scaled)[0]
        future_prices.append(predicted_price)

    future_df['Forecasted_Price'] = future_prices  # Store predictions

    import matplotlib.dates as mdates
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot historical prices
    plt.plot(df['Date'], df['Price'], label='Historical Prices', color='#1f77b4', linewidth=2)
    
    # Plot future predictions
    plt.plot(future_dates, future_df['Forecasted_Price'], label='Random Forest Forecast', linestyle='--', color='#ff5733', linewidth=2)
    
    # Set title and labels
    plt.title('Random Forest - Future Market Price Prediction', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Predicted Price (INR)', fontsize=14)
    
    # Rotate date labels for better readability
    plt.xticks(rotation=45)
    
    # Add gridlines for better visual clarity
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis to display dates properly
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Add legend
    plt.legend(loc='upper left', fontsize=12)
    
    # Show plot
    plt.tight_layout()  # Adjust layout to avoid cutting off content
    plt.show()

    return {
    "rmse": rmse,
    "mae": mae,
    "mape": mape,
    "model_path": "random_forest_model.pkl",
    "scaler_path": "scaler.pkl"
}
