# Gated recurrent unit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from evaluation import evaluate_model

def train_gru(df):
    # ✅ Normalize price data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Price']])

    # ✅ Split into train & test
    train_size = int(len(df) * 0.8)
    train, test = df_scaled[:train_size], df_scaled[train_size:]

    # ✅ Sequence creation function
    def create_sequences(data, seq_length=10):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y).reshape(-1, 1)  # ✅ Ensure y is 2D

    # ✅ Create train/test sequences
    X_train, y_train = create_sequences(train)
    X_test, y_test = create_sequences(test)

    # ✅ Reshape X to match GRU input requirements
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # ✅ Build GRU model
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        GRU(50),
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')

    # ✅ Train model
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

    # ✅ Predict test values
    gru_forecast = model.predict(X_test)
    gru_forecast = scaler.inverse_transform(gru_forecast).flatten()  # ✅ Flatten for correct shape

    # ✅ Ensure y_true matches predicted shape
    y_true = df['Price'][train_size+10:].values.flatten()

    # ✅ Evaluate model performance
    rmse, mae, mape = evaluate_model(y_true, gru_forecast, "GRU")

    # ✅ Generate future forecasts for 30 days
    future_input = test[-10:].reshape((1, 10, 1))  # Start with last 10 days
    future_forecast = []

    for _ in range(30):
        next_pred = model.predict(future_input)  # Predict next day
        future_forecast.append(next_pred[0, 0])  # Store prediction

        # Update future_input (remove oldest, append new prediction)
        future_input = np.roll(future_input, shift=-1, axis=1)
        future_input[0, -1, 0] = next_pred[0, 0]

    # ✅ Convert forecast to original scale
    future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1)).flatten()

    # ✅ Generate future dates
    future_dates = pd.date_range(start=df['Date'].max(), periods=30, freq='D')

    # ✅ Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Price'], label='Historical Prices', color='blue')
    plt.plot(df['Date'][train_size+10:], gru_forecast, label='Test Predictions', linestyle='dashed', color='green')
    plt.plot(future_dates, future_forecast, label='Future Forecast (30 days)', linestyle='dashed', color='red')
    plt.title('GRU Model Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return rmse, mae, mape
