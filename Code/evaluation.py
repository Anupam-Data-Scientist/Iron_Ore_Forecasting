from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(y_true, y_pred, model_name):
    """Evaluates the model using RMSE, MAE, MAPE, and R-squared."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    # r2 = r2_score(y_true, y_pred)

    print(f'{model_name} Evaluation:')
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print(f'MAPE: {mape}')
    # print(f'R-squared: {r2}')
    
    return rmse, mae, mape

def compare_models(results):
    # Get the model with the lowest Test RMSE
    best_model = min(results, key=lambda x: results[x]['Test RMSE'])  # Choose model with lowest Test RMSE
    print(f"🏆 Best Model: {best_model} with RMSE: {results[best_model]['Test RMSE']:.4f}")

    model_names = list(results.keys())
    rmse_values = [results[m]['Test RMSE'] for m in model_names]  # Using 'Test RMSE'
    mae_values = [results[m]['Test MAE'] for m in model_names]  # Using 'Test MAE'
    mape_values = [results[m]['Test MAPE'] for m in model_names]  # Using 'Test MAPE'

    x = np.arange(len(model_names))  # Model indices

    plt.figure(figsize=(12, 5))
    plt.bar(x - 0.2, rmse_values, 0.2, label="RMSE")
    plt.bar(x, mae_values, 0.2, label="MAE")
    plt.bar(x + 0.2, mape_values, 0.2, label="MAPE")
    plt.xticks(x, model_names, rotation=45)
    plt.ylabel("Error Value")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.show()

