""" EWMA Prediction Method

This module contains the Exponentially Weighted Moving Average Prediction method.
It is a wrapper around the statsmodels library.

"""

from typing import TypeVar, Callable
from data.Data import Dataset, Result
from predictions.Prediction import PredictionData
import logging

Data = TypeVar("Data", contravariant=True)
Model = TypeVar("Model")
Prediction = TypeVar("Prediction")

from .predict import Predict


def ses(
    seasonal_decompose_data: Callable[[Dataset], Dataset],
    fit_model: Callable[[Dataset], Model],
    forecast: Callable[[Model, Dataset, Dataset], Prediction],
) -> Predict[Dataset, PredictionData]:
    def predict(
        data: Dataset,
    ) -> Prediction:
        decomposed_dataset = seasonal_decompose_data(data)
        trained_model = fit_model(decomposed_dataset)
        return forecast(trained_model, decomposed_dataset, data)

    return predict