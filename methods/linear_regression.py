""" Linear regression 
    A simple machine learning algorithm that 
    fits a linear model to the data."""

from typing import TypeVar, Callable

Data = TypeVar("Data", contravariant=True)
Model = TypeVar("Model")
Prediction = TypeVar("Prediction")

from .predict import Predict

def linear_regression(
    stationarity: Callable[[Data], bool],
    fit_linear_regression_model: Callable[[Data], Model],
    forecast: Callable[[Model, int], Prediction],
) -> Predict[Data, Prediction]:
    def predict(
        data: Data,
    ) -> Prediction:
        if stationarity(data):
            return
        trained_model = fit_linear_regression_model(data)
        return forecast(trained_model, 1)
    return predict