import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.utils import evaluate_model


def simple_average_ensemble(predictions_list, y_true):
    """Equal weight average of all model predictions."""
    avg_pred = np.mean(predictions_list, axis=0)
    results = evaluate_model('Simple Average Ensemble', y_true, avg_pred)
    return avg_pred, results


def weighted_average_ensemble(predictions_list, y_true):
    """Inverse-MAE weighted average ensemble."""
    from sklearn.metrics import mean_absolute_error
    maes = [mean_absolute_error(y_true, p) for p in predictions_list]
    inv_maes = np.array([1/m for m in maes])
    weights = inv_maes / inv_maes.sum()
    print(f"Weights: {[round(w, 4) for w in weights]}")
    weighted_pred = sum(w * p for w, p in zip(weights, predictions_list))
    results = evaluate_model('Weighted Average Ensemble', y_true, weighted_pred)
    return weighted_pred, weights, results


def stacking_ensemble(predictions_list, y_true, meta_split=0.5):
    """Ridge regression stacking ensemble."""
    stacked = np.column_stack(predictions_list)
    split   = int(len(y_true) * meta_split)
    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(stacked[:split], y_true[:split])
    stacking_pred = meta_learner.predict(stacked[split:])
    results = evaluate_model('Stacking Ensemble',
                              y_true[split:], stacking_pred)
    return stacking_pred, meta_learner, results