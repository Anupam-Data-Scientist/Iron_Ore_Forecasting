import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from evaluation import evaluate_model
import joblib
from tensorflow.keras.utils import get_custom_objects


# Custom Transformer model class
class TransformerTimeSeries(tf.keras.Model):
    def __init__(self, input_shape, **kwargs):
        super(TransformerTimeSeries, self).__init__(**kwargs)
        self.encoder_layer = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)
        self.input_shape = input_shape  # Save the input shape as an attribute

    def call(self, inputs):
        x = self.encoder_layer(inputs, inputs)
        x = tf.reduce_mean(x, axis=1)
        x = self.dense1(x)
        return self.dense2(x)

    # Define get_config method to store the layer configuration
    def get_config(self):
        config = super(TransformerTimeSeries, self).get_config()  # Get the default config
        config.update({"input_shape": self.input_shape})  # Add input_shape to the config
        return config

    # Define from_config method to create an instance from the configuration
    @classmethod
    def from_config(cls, config):
        input_shape = config.pop('input_shape')  # Pop the input_shape from the config
        return cls(input_shape=input_shape, **config)  # Pass it during initialization


# Register the custom model with Keras
get_custom_objects()['TransformerTimeSeries'] = TransformerTimeSeries


# Function to create sequences for training/testing
def create_sequences(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y).reshape(-1, 1)


# Function to train the transformer model
def train_transformer(df):
    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0,1))
    df_scaled = scaler.fit_transform(df[['Price']])

    # Splitting the data into train and test sets
    train_size = int(len(df) * 0.8)
    train, test = df_scaled[:train_size], df_scaled[train_size:]

    # Creating sequences for training and testing
    X_train, y_train = create_sequences(train)
    X_test, y_test = create_sequences(test)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Creating and training the model
    model = TransformerTimeSeries(input_shape=(X_train.shape[1], 1))  # Pass the input shape here
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

    # Forecasting on the test set
    transformer_forecast = model.predict(X_test)
    transformer_forecast = scaler.inverse_transform(transformer_forecast).flatten()
    y_true = df['Price'][train_size+10:].values.flatten()

    # Evaluate the model
    rmse, mae, mape = evaluate_model(y_true, transformer_forecast, "Transformer")

    # Future forecasting (next 30 days)
    future_forecast = []
    last_sequence = test[-10:].reshape(1, 10, 1)
    for _ in range(30):
        next_pred = model.predict(last_sequence)
        future_forecast.append(next_pred[0, 0])
        last_sequence = np.append(last_sequence[:, 1:, :], [[[next_pred[0, 0]]]], axis=1)
    
    future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(start=df['Date'].max(), periods=30, freq='D')

    # Plotting the results
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Price'], label='Historical Prices', color='blue')
    plt.plot(df['Date'][train_size+10:], transformer_forecast, label='Test Predictions', linestyle='dashed', color='green')
    plt.plot(future_dates, future_forecast, label='Future Forecast (30 days)', linestyle='dashed', color='red')
    plt.title('Transformer Model Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Save the model in SavedModel format
    model.save('transformer_model')  # This saves the model as a directory (no .h5, just saved_model.pb)
    joblib.dump(scaler, 'scaler.pkl')  # Save the scaler

    return rmse, mae, mape
