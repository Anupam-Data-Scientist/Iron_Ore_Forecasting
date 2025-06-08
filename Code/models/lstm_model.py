import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from evaluation import evaluate_model
import joblib

def train_lstm(df):
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Price']])

    # Split the data into train and test sets
    train_size = int(len(df) * 0.8)
    train, test = df_scaled[:train_size], df_scaled[train_size:]

    def create_sequences(data, seq_length=10):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y).reshape(-1, 1)

    # Create sequences for training and testing
    X_train, y_train = create_sequences(train)
    X_test, y_test = create_sequences(test)

    # Ensure the correct shape for LSTM: (samples, time steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Define the LSTM model
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),  # Dropout layer to reduce overfitting
        LSTM(100),
        Dropout(0.2),  # Dropout layer to reduce overfitting
        Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')

    # Early stopping to avoid overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Predict on the test set
    lstm_forecast = model.predict(X_test)
    lstm_forecast = scaler.inverse_transform(lstm_forecast).flatten()

    # True values for evaluation
    y_true = df['Price'][train_size + 10:].values.flatten()

    # Evaluate the model
    rmse, mae, mape = evaluate_model(y_true, lstm_forecast, "LSTM")

    # Future forecast (30 days)
    future_forecast = []
    last_sequence = test[-10:].reshape(1, 10, 1)  # Use last 10 data points to predict future values

    for _ in range(30):
        next_pred = model.predict(last_sequence)
        future_forecast.append(next_pred[0, 0])  # Scalar value
        last_sequence = np.append(last_sequence[:, 1:, :], [[[next_pred[0, 0]]]], axis=1)

    future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1)).flatten()

    # Generate future dates
    future_dates = pd.date_range(start=df['Date'].max(), periods=30, freq='D')

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Price'], label='Historical Prices', color='blue')
    plt.plot(df['Date'][train_size + 10:], lstm_forecast, label='Test Predictions', linestyle='dashed', color='green')
    plt.plot(future_dates, future_forecast, label='Future Forecast (30 days)', linestyle='dashed', color='red')
    plt.title('LSTM Model Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Save the model and scaler for deployment
    joblib.dump(model, 'lstm_model.pkl')  # Save the trained LSTM model
    joblib.dump(scaler, 'scaler.pkl')  # Save the scaler for future transformations

    return rmse, mae, mape

