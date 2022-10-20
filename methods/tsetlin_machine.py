"""
Tsetlin Machine Regression method

This module contains the Tsetlin Machine Regression method.
It is a wrapper around the pyTsetlinMachine library.

"""

from typing import Tuple, TypeVar, Callable, Dict
from data.dataset import Dataset
from predictions.Prediction import PredictionData
import logging

from .predict import Predict

Data = TypeVar("Data", contravariant=True)
Model = TypeVar("Model")
Prediction = TypeVar("Prediction")

def tsetlin_machine(
    get_best_model: Callable[[Dataset, bool], Tuple[Model, dict]],
    get_forecast: Callable[[Dataset, Model, dict], Prediction],
) -> Predict[Dataset, PredictionData]:
    """
    Return a function that takes a dataset and returns a prediction.
    """
    def predict(dataset: Dataset, parallel: bool = False) -> PredictionData:
        """
        Return a prediction for the given dataset.
        """
        logging.info("Tsetlin Machine Regression")
        model, params = get_best_model(dataset, parallel)
        return get_forecast(dataset, model, params)
    return predict




