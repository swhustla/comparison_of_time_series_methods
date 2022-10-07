""" Facebook Prophet method for time series forecasting. 

    This module contains the Prophet method for time series forecasting.
    
    It is a wrapper around the Prophet library.
"""

from typing import TypeVar, Callable
from data.dataset import Dataset
from predictions.Prediction import PredictionData

Data = TypeVar("Data", contravariant=True)
Model = TypeVar("Model")
Prediction = TypeVar("Prediction")

from .predict import Predict


def prophet(
    fit_prophet_model: Callable[[Dataset], Model],
    forecast: Callable[[Model, Dataset], PredictionData],
) -> Predict[Dataset, PredictionData]:
    def predict(
        data: Dataset,
    ) -> Prediction:
        trained_model = fit_prophet_model(data)
        return forecast(trained_model, data)

    return predict
    