"""Kalman Filter for 1D data.


This module contains the Kalman Filter Prediction method.
The Kalman Filter is a state space model that is used to estimate the state of a system.
It is a wrapper around the pykalman library.
"""

from typing import TypeVar, Callable
from data.dataset import Dataset
from predictions.Prediction import PredictionData
import logging

Data = TypeVar("Data", contravariant=True)
Model = TypeVar("Model")
Prediction = TypeVar("Prediction")


from .predict import Predict


def kalman_filter(
    fit_kalman_filter_model: Callable[[Dataset], Model],
    forecast: Callable[[Model, Dataset], PredictionData],
) -> Predict[Dataset, PredictionData]:
    def predict(
        data: Dataset,
    ) -> Prediction:
        trained_model = fit_kalman_filter_model(data)
        return forecast(trained_model, data)

    return predict
