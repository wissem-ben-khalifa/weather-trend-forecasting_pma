import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from src.utils import make_time_features, evaluate_model


def train_linear_regression(X_train, y_train, X_test, y_test):
    """Train and evaluate Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results = evaluate_model('Linear Regression', y_test, preds)
    return model, preds, results


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and evaluate XGBoost model."""
    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05,
        max_depth=5, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    results = evaluate_model('XGBoost', y_test, preds)
    return model, preds, results


def train_arima(train_series, test_series, order=(5,1,2)):
    """Train and evaluate ARIMA model."""
    model = ARIMA(train_series, order=order).fit()
    preds = model.forecast(steps=len(test_series)).values
    results = evaluate_model(f'ARIMA{order}',
                              test_series.values, preds)
    return model, preds, results


def prepare_forecast_data(daily_temp, split_date):
    """Prepare train/test splits for forecasting."""
    from src.utils import make_time_features
    train = daily_temp[daily_temp.index < split_date]
    test  = daily_temp[daily_temp.index >= split_date]
    full_feat  = make_time_features(daily_temp)
    split_idx  = full_feat.index.searchsorted(split_date)
    X = full_feat.drop('y', axis=1)
    y = full_feat['y']
    return (train, test,
            X.iloc[:split_idx], y.iloc[:split_idx],
            X.iloc[split_idx:],  y.iloc[split_idx:])