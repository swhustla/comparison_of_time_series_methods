import numpy as np
from typing import Dict

from methods.get_metrics import get_metrics as method

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

def __round_to_4dp(value: float) -> float:
    return round(value, 4)

def __metrics(ground_truth: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    return dict(map(lambda metric: (metric, __round_to_4dp(__dict_of_metrics[metric](ground_truth, prediction))), __dict_of_metrics))


get_metrics = method(__metrics)