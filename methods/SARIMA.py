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



def sarima(
    get_best_sarima_model: Callable[[Dataset], Model],
    forecast: Callable[[Model, int], Prediction]
) -> Predict[Dataset, PredictionData]:
    """
    Returns a SARIMA prediction method.
    """
    def predict(dataset: Dataset) -> PredictionData:
        """
        Predict the given dataset using the SARIMA method.
        """
        logging.info("Starting SARIMA prediction")
        model, number_of_configs = get_best_sarima_model(dataset)
        prediction = forecast(model, dataset, number_of_configs)
        logging.info("Finished SARIMA prediction")
        return prediction

    return predict

