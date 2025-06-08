import optuna
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def preprocess_data(df):
    """ Adds lag features and drops missing values. """
    df = df.copy()
    df['Price_Lag1'] = df['Price'].shift(1)
    df['Price_Lag7'] = df['Price'].shift(7)
    df['Price_Lag14'] = df['Price'].shift(14)  # New lag feature
    df['Rolling_Avg_7'] = df['Price'].rolling(window=7).mean()
    df.dropna(inplace=True)  # Remove NaN rows created by shift
    return df

def objective(trial, df):
    """ Optuna objective function for tuning XGBoost. """
    df = preprocess_data(df)

    features = ['Vol.', 'Change %', 'SMA_30', 'EMA_30', 'month', 'quarter', 'day_of_week', 
                'Price_Lag1', 'Price_Lag7', 'Price_Lag14', 'Rolling_Avg_7']
    X = df[features]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    params = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.06),  # Keep it low
        'max_depth': trial.suggest_int('max_depth', 3, 6),  # Reduce complexity
        'subsample': trial.suggest_float('subsample', 0.5, 0.7),  # Prevent overfitting
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.85),
        'alpha': trial.suggest_float('alpha', 0.5, 1.5),  # Increase L1 regularization
        'lambda': trial.suggest_float('lambda', 1.5, 2.5),  # Stronger L2 regularization
        'min_child_weight': trial.suggest_int('min_child_weight', 4, 6),  # Prevents small splits
        'random_state': 42
    }


    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        early_stopping_rounds=10,
        verbose=False
    )
    y_pred = model.predict(X_test)
    
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, 1))) * 100  # Avoid divide by zero error 

    return rmse, mae, mape

def tune_xgboost(df):
    """ Uses Optuna to find the best XGBoost hyperparameters and evaluate model performance. """
    df = preprocess_data(df)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, df)[0], n_trials=100)  # Optimize based on RMSE

    print("Best parameters:", study.best_params)

    # Get final evaluation scores using the best parameters
    best_rmse, best_mae, best_mape = objective(study.best_trial, df)

    print(f"Final XGBoost Performance - RMSE: {best_rmse:.4f}, MAE: {best_mae:.4f}, MAPE: {best_mape:.2f}%")

    return study.best_params, best_rmse, best_mae, best_mape
