""" ARIMA Prediction 

    This module contains the Auto Regressive Moving Average Prediction method.
    It is a wrapper around the pmdarima and statsmodels libraries.
"""

from typing import TypeVar, Callable
from data.dataset import Dataset
from predictions.Prediction import PredictionData
import logging


Data = TypeVar("Data", contravariant=True)
Model = TypeVar("Model")
Prediction = TypeVar("Prediction")


from .predict import Predict


def arima(
    fit_auto_regressive_model: Callable[[Dataset], Model],
    forecast: Callable[[Model, Dataset], PredictionData],
) -> Predict[Dataset, PredictionData]:
    def predict(
        data: Dataset,
    ) -> Prediction:
        trained_model = fit_auto_regressive_model(data)
        return forecast(trained_model, data)

    return predict
