"""SARIMA Prediction

This module contains the SARIMA prediction method.
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



def auto_arima(
    fit_pmdarima_model: Callable[[Dataset], Model],
    forecast_pmdarima: Callable[[Model, Dataset], Prediction]
) -> Predict[Dataset, PredictionData]:
    """
    Returns a SARIMA prediction method via auto arima.
    """
    def predict(dataset: Dataset) -> PredictionData:
        """
        Predict the given dataset using the SARIMA method.
        """
        logging.info("Starting SARIMA (auto arima) prediction")
        model = fit_pmdarima_model(dataset)
        prediction = forecast_pmdarima(model, dataset)
        logging.info("Finished SARIMA (auto arima) prediction")
        return prediction

    return predict

