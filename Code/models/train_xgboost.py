import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from evaluation import evaluate_model

def train_xgboost(df, best_params):
    df = df.copy()

    # Creating Lag Features
    df['Price_Lag1'] = df['Price'].shift(1)
    df['Price_Lag7'] = df['Price'].shift(7)

    # Defining Feature Set
    features = ['Vol.', 'Change %', 'SMA_30', 'EMA_30', 'month', 'quarter', 'day_of_week', 'Price_Lag1', 'Price_Lag7']
    df.dropna(inplace=True)

    # Scaling the Features
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    # Splitting Data
    X = df[features]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Training the Model with Tuned Parameters
    model = XGBRegressor(
        n_estimators=best_params['n_estimators'], 
        learning_rate=best_params['learning_rate'], 
        max_depth=min(best_params['max_depth'], 6),
        subsample=max(best_params['subsample'], 0.7),
        colsample_bytree=max(best_params['colsample_bytree'], 0.7),
        reg_alpha=best_params.get('reg_alpha', 0.1),
        reg_lambda=best_params.get('reg_lambda', 1.0),
        random_state=42
    )
    model.fit(X_train, y_train, 
              eval_set=[(X_train, y_train), (X_test, y_test)], 
              early_stopping_rounds=20, verbose=False)

    # 🔹 Predictions on Training & Testing Data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 🔹 Model Evaluation
    train_rmse, train_mae, train_mape = evaluate_model(y_train, y_train_pred, "XGBoost (Training)")
    test_rmse, test_mae, test_mape = evaluate_model(y_test, y_test_pred, "XGBoost (Testing)")

    # 🔹 Save the trained model
    with open("xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # 🔹 Save the scaler (to ensure consistent scaling in Streamlit)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("✅ Model and Scaler Saved!")

    # 🔹 Visualizing Training vs. Test Predictions
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_train)), y_train.values, label="Actual Train", color='blue')
    plt.plot(range(len(y_train)), y_train_pred, label="Predicted Train", linestyle='dashed', color='red')
    plt.title('XGBoost - Training Data Comparison')
    plt.xlabel('Samples')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_test)), y_test.values, label="Actual Test", color='blue')
    plt.plot(range(len(y_test)), y_test_pred, label="Predicted Test", linestyle='dashed', color='red')
    plt.title('XGBoost - Testing Data Comparison')
    plt.xlabel('Samples')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # 🔹 Future Predictions
    future_dates = pd.date_range(start=df['Date'].max(), periods=30, freq='D')
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['month'] = future_df['Date'].dt.month
    future_df['quarter'] = future_df['Date'].dt.quarter
    future_df['day_of_week'] = future_df['Date'].dt.dayofweek
    future_df['SMA_30'] = df['SMA_30'].iloc[-1]
    future_df['EMA_30'] = df['EMA_30'].iloc[-1]
    future_df['Vol.'] = df['Vol.'].iloc[-1]
    future_df['Change %'] = np.random.uniform(-0.02, 0.02, 30)
    future_df['Price_Lag1'] = df['Price'].iloc[-1]
    future_df['Price_Lag7'] = df['Price'].iloc[-7] if len(df) > 7 else df['Price'].iloc[-1]
    
    future_X_scaled = pd.DataFrame(scaler.transform(future_df[features]), columns=features)
    future_predictions = model.predict(future_X_scaled)

    # 🔹 Visualization for Future Prediction
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Price'], label='Historical Prices', color='blue')
    plt.plot(future_dates, future_predictions, label='XGBoost Forecast', linestyle='dashed', color='red')
    plt.title('XGBoost - Future Market Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price')
    plt.legend()
    plt.show()

    # 🔹 Return trained model and evaluation metrics
    return model, {
        "Train RMSE": train_rmse, "Test RMSE": test_rmse,
        "Train MAE": train_mae, "Test MAE": test_mae,
        "Train MAPE": train_mape, "Test MAPE": test_mape
    }
