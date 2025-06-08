import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from evaluation import evaluate_model

def train_svr(df, best_params):
    df = df.copy()
    
    # Feature Engineering
    df['Price_Lag1'] = df['Price'].shift(1)
    df['Price_Lag7'] = df['Price'].shift(7)
    df.dropna(inplace=True)
    
    features = ['Vol.', 'Change %', 'SMA_30', 'EMA_30', 'month', 'quarter', 'day_of_week', 'Price_Lag1', 'Price_Lag7']
    
    # Scaling
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Train-test split
    X = df[features]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model with Tuned Parameters
    model = SVR(
        kernel=best_params['kernel'], 
        C=best_params['C'], 
        gamma=best_params['gamma'], 
        epsilon=best_params['epsilon']
    )
    model.fit(X_train, y_train)
    
    # Predictions & Evaluation
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_rmse, train_mae, train_mape = evaluate_model(y_train, y_train_pred, "SVR - Training")
    test_rmse, test_mae, test_mape = evaluate_model(y_test, y_test_pred, "SVR - Testing")
    
    # 🔹 Visualizing Training vs. Test Predictions
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_train)), y_train.values, label="Actual Train", color='blue')
    plt.plot(range(len(y_train)), y_train_pred, label="Predicted Train", linestyle='dashed', color='red')
    plt.title('SVR - Training Data Comparison')
    plt.xlabel('Samples')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_test)), y_test.values, label="Actual Test", color='blue')
    plt.plot(range(len(y_test)), y_test_pred, label="Predicted Test", linestyle='dashed', color='red')
    plt.title('SVR - Testing Data Comparison')
    plt.xlabel('Samples')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    
    # Future Predictions (30 Days)
    future_dates = pd.date_range(start=df['Date'].max(), periods=30, freq='D')
    future_df = pd.DataFrame({'Date': future_dates})
    
    # Generate synthetic future values
    future_df['month'] = future_df['Date'].dt.month
    future_df['quarter'] = future_df['Date'].dt.quarter
    future_df['day_of_week'] = future_df['Date'].dt.dayofweek
    future_df['SMA_30'] = df['SMA_30'].iloc[-1] * (1 + np.linspace(0, 0.02, 30))
    future_df['EMA_30'] = df['EMA_30'].iloc[-1] * (1 + np.linspace(0, 0.015, 30))
    future_df['Vol.'] = df['Vol.'].iloc[-1] * (1 + np.linspace(0, 0.03, 30))
    future_df['Change %'] = np.random.uniform(-0.02, 0.02, 30)
    
    # Add lag features (use last available values)
    future_df['Price_Lag1'] = df['Price'].iloc[-1]
    future_df['Price_Lag7'] = df['Price'].iloc[-7] if len(df) > 7 else df['Price'].iloc[-1]
    
    # Apply scaling
    future_X_scaled = pd.DataFrame(scaler.transform(future_df[features]), columns=features)
    
    # Make predictions
    future_predictions = model.predict(future_X_scaled)
    
    # Plot Future Predictions
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Price'], label='Historical Prices', color='blue')
    plt.plot(future_dates, future_predictions, label='SVR Forecast', linestyle='dashed', color='red')
    plt.title('SVR - Future Market Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.show()
    
    # 🔹 Return both training and testing metrics
    return {
        "Train RMSE": train_rmse, "Test RMSE": test_rmse,
        "Train MAE": train_mae, "Test MAE": test_mae,
        "Train MAPE": train_mape, "Test MAPE": test_mape
    }
