"""Holt-Winters Exponential Smoothing forecast

This module contains the Holt-Winters Exponential Smoothing forecast method.
It is a wrapper around the statsmodels library.

"""


from typing import Tuple, TypeVar, Callable, Dict
from data.dataset import Dataset
from predictions.Prediction import PredictionData
import logging

from .predict import Predict

Data = TypeVar("Data", contravariant=True)
Model = TypeVar("Model")
Prediction = TypeVar("Prediction")

__parallel = False

def holt_winters(
    get_best_model: Callable[[Dataset, bool], Tuple[Model, dict, int]],
    get_forecast: Callable[[Dataset, Model, dict, int], Prediction],
) -> Predict[Dataset, PredictionData]:
    """
    Return a function that takes a dataset and returns a prediction.
    """
    def predict(dataset: Dataset, parallel: bool = False) -> PredictionData:
        """
        Return a prediction for the given dataset.
        """
        logging.info("Holt-Winters Exponential Smoothing forecast")
        model, params, number_of_configs = get_best_model(dataset, parallel)
        return get_forecast(dataset, model, params, number_of_configs)
    return predict
