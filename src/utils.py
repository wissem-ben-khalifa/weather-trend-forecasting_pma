import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(name, y_true, y_pred):
    """Evaluate a regression model and return metrics dict."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
    print(f"\n{'='*40}")
    print(f"Model: {name}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    return {
        'Model': name,
        'MAE':   round(mae,  4),
        'RMSE':  round(rmse, 4),
        'R2':    round(r2,   4),
        'MAPE':  round(mape, 2)
    }


def make_time_features(series):
    """Generate lag and rolling features for time series forecasting."""
    df = pd.DataFrame({'y': series})
    df['lag_1']           = df['y'].shift(1)
    df['lag_7']           = df['y'].shift(7)
    df['lag_14']          = df['y'].shift(14)
    df['lag_30']          = df['y'].shift(30)
    df['rolling_mean_7']  = df['y'].rolling(7).mean()
    df['rolling_mean_30'] = df['y'].rolling(30).mean()
    df['rolling_std_7']   = df['y'].rolling(7).std()
    df['month']           = series.index.month
    df['day_of_year']     = series.index.dayofyear
    df['day_of_week']     = series.index.dayofweek
    return df.dropna()


def classify_climate_zone(lat):
    """Classify latitude into climate zone."""
    lat = abs(lat)
    if lat <= 23.5:
        return 'Tropical (0-23.5°)'
    elif lat <= 35:
        return 'Subtropical (23.5-35°)'
    elif lat <= 60:
        return 'Temperate (35-60°)'
    else:
        return 'Polar (60-90°)'


def normalize_series(series):
    """Normalize a pandas Series to 0-1 range."""
    return (series - series.min()) / (series.max() - series.min())