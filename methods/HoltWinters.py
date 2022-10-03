"""Holt-Winters Exponential Smoothing forecast

This module contains the Holt-Winters Exponential Smoothing forecast method.
It is a wrapper around the statsmodels library.

"""


from typing import Tuple, TypeVar, Callable, Dict
from data.Data import Dataset, Result
from predictions.Prediction import PredictionData
import logging

from .predict import Predict

Data = TypeVar("Data", contravariant=True)
Model = TypeVar("Model")
Prediction = TypeVar("Prediction")

__parallel = False

def holt_winters(
    get_best_model: Callable[[Dataset, bool], Tuple[Model, dict]],
    get_forecast: Callable[[Dataset, Model, dict], Prediction],
) -> Predict[Dataset, Result]:
    """
    Return a function that takes a dataset and returns a prediction.
    """
    def predict(dataset: Dataset, parallel: bool = False) -> PredictionData:
        """
        Return a prediction for the given dataset.
        """
        logging.info("Holt-Winters Exponential Smoothing forecast")
        model, params = get_best_model(dataset, parallel)
        return get_forecast(dataset, model, params)
    return predict
