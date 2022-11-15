"""Simple moving average method.

This module contains the Simple Moving Average Prediction method.
It is a wrapper around the statsmodels library.

"""

from typing import TypeVar, Callable
from data.dataset import Dataset
from predictions.Prediction import PredictionData
import logging

Data = TypeVar("Data", contravariant=True)
Model = TypeVar("Model")
Prediction = TypeVar("Prediction")


from .predict import Predict

__ma_order = 1

def ma(
    fit_simple_ma: Callable[[Dataset], Model],
    forecast: Callable[[Model, Dataset], PredictionData],
) -> Predict[Dataset, PredictionData]:
    """
    Create a simple moving average prediction method.
    """

    def predict(data: Data) -> Prediction:
        """
        Predict using the simple moving average method.
        """
        model = fit_simple_ma(data, __ma_order)
        return forecast(model, data)

    return predict
