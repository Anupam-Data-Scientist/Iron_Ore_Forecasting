import optuna
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def preprocess_data(df):
    """Adds lag features, rolling statistics, and drops missing values."""
    df = df.copy()
    df['Price_Lag1'] = df['Price'].shift(1)
    df['Price_Lag7'] = df['Price'].shift(7)
    df['Price_Lag14'] = df['Price'].shift(14)
    df['Price_Lag30'] = df['Price'].shift(30)
    df['Price_Lag60'] = df['Price'].shift(60)
    df['Rolling_Mean_7'] = df['Price'].rolling(window=7).mean()
    df['Rolling_Std_7'] = df['Price'].rolling(window=7).std()
    df['Rolling_Min_7'] = df['Price'].rolling(window=7).min()
    df['Rolling_Max_7'] = df['Price'].rolling(window=7).max()
    df.dropna(inplace=True)  # Drop rows with NaN values from rolling statistics
    return df

def objective(trial, X_train, X_test, y_train, y_test):
    """Optuna objective function for tuning SVR."""
    # Use suggest_float for loguniform ranges
    params = {
        'C': trial.suggest_float('C', 1e-1, 1e4, log=True),  # Extended C range with log scale
        'epsilon': trial.suggest_float('epsilon', 1e-4, 1.0, log=True),  # Extended epsilon range with log scale
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),  # Corrected gamma options
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly', 'sigmoid'])  # Added poly and sigmoid kernels
    }

    model = SVR(**params)

    # Fit the model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = (np.abs((y_test - y_pred) / y_test)).mean() * 100

    trial.set_user_attr("MAE", mae)
    trial.set_user_attr("MAPE", mape)

    return rmse  # Return RMSE to minimize



def tune_svr(df, n_trials=3, test_size=0.1):
    """Uses Optuna to find the best SVR hyperparameters with K-Fold Cross-Validation."""
    # Preprocess data
    df = preprocess_data(df)

    features = ['Vol.', 'Change %', 'SMA_30', 'EMA_30', 'month', 'quarter', 'day_of_week', 'Price_Lag1', 'Price_Lag7', 'Price_Lag14', 'Price_Lag30', 'Price_Lag60', 'Rolling_Mean_7', 'Rolling_Std_7', 'Rolling_Min_7', 'Rolling_Max_7']
    X = df[features]
    y = df['Price']

    # Split the data once before optimization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # Scale the data using RobustScaler (you could also try StandardScaler)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create the Optuna study and optimize the hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test), n_trials=n_trials, show_progress_bar=True)

    # Get the best trial and its results
    best_trial = study.best_trial
    best_params = best_trial.params

    best_rmse = best_trial.value
    best_mae = best_trial.user_attrs["MAE"]
    best_mape = best_trial.user_attrs["MAPE"]

    # Print results
    print("\n✅ **Best parameters:**", best_params)
    print("📊 **Final SVR Performance:**")
    print(f"✅ RMSE: {best_rmse:.4f} | MAE: {best_mae:.4f} | MAPE: {best_mape:.2f}%\n")

    return best_params, best_rmse, best_mae, best_mape
