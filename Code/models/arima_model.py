import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from evaluation import evaluate_model

def train_arima(df):
    train_size = int(len(df) * 0.8)
    train, test = df['Price'][:train_size], df['Price'][train_size:]

    model = ARIMA(train, order=(5,1,0))
    arima_fit = model.fit()

    arima_forecast = arima_fit.forecast(steps=len(test))

    rmse, mae, mape = evaluate_model(test, arima_forecast, "ARIMA")

    future_forecast = arima_fit.forecast(steps=30)

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Price'], label='Historical Prices')
    plt.plot(df['Date'][train_size:], arima_forecast, label='Test Predictions', linestyle='dashed', color='green')
    plt.plot(pd.date_range(start=df['Date'].max(), periods=30, freq='D'), future_forecast, 
             label='Future Forecast (30 days)', linestyle='dashed', color='red')
    plt.title('ARIMA Model Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    return rmse, mae, mape
