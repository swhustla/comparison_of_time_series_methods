import numpy as np
from typing import Dict

from methods.get_metrics import get_metrics as method

from predictions.Prediction import PredictionData

#TODO: Add time taken to metrics

def __get_root_mean_squared_error(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    return np.sqrt(np.mean((ground_truth - prediction) ** 2))

def __get_r_squared(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    return 1 - (np.sum((ground_truth - prediction) ** 2) / np.sum((ground_truth - np.mean(ground_truth)) ** 2))

def __get_mean_absolute_error(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    return np.mean(np.abs(ground_truth - prediction))

__dict_of_metrics = {
        "root_mean_squared_error": __get_root_mean_squared_error,
        "r_squared": __get_r_squared,
        "mean_absolute_error": __get_mean_absolute_error,
    }

def __round_to_2dp(value: float) -> float:
    return round(value, 2)

def __metrics(prediction: PredictionData) -> Dict[str, float]:
    if type(prediction.ground_truth_values) == np.ndarray:
        ground_truth = prediction.ground_truth_values
    else:
        ground_truth = prediction.ground_truth_values.values
    if prediction.prediction_column_name is not None:
        prediction = prediction.values[prediction.prediction_column_name].values
    else:
        prediction = prediction.values
    return dict(map(lambda metric: (metric, __round_to_2dp(__dict_of_metrics[metric](ground_truth, prediction))), __dict_of_metrics))


get_metrics = method(__metrics)