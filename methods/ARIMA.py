""" ARIMA Prediction 

    This module contains the Auto Regressive Moving Average Prediction method.
    It is a wrapper around the pmdarima and statsmodels libraries.
"""

from typing import TypeVar, Callable


Data = TypeVar("Data", contravariant=True)
Model = TypeVar("Model")
Prediction = TypeVar("Prediction")


from .predict import Predict


def arima(
    stationarity: Callable[[Data], bool],
    fit_auto_regressive_model: Callable[[Data, int], Model],
    number_of_steps: Callable[[Data], int],
    forecast: Callable[[Model, int], Prediction],
) -> Predict[Data, Prediction]:
    def predict(
        data: Data,
    ) -> Prediction:
        if stationarity(data):
            return
        trained_model = fit_auto_regressive_model(data)
        return forecast(trained_model, number_of_steps(data))

    return predict
