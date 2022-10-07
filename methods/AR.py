"""
Auto Regressive Prediction


This module contains the Auto Regressive Prediction method.
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


def ar(
    quick_check_for_auto_correlation: Callable[[Dataset], bool],
    train_auto_regressive_model: Callable[[Dataset], Model],
    forecast: Callable[[Model, Dataset], Prediction],
) -> Predict[Dataset, Prediction]:
    """
    Auto Regressive Prediction
    function that predicts the next 20% of the data using the Auto Regressive method.
    """

    def predict(data: Dataset) -> Prediction:
        """
        Predicts the next 20% of the data using the Auto Regressive method.

        :param data: The data to predict.
        :return: The predicted data.
        """
        logging.info(f"Predicting {data.name} using Auto Regressive Prediction")
        if quick_check_for_auto_correlation(data):
            logging.info(f"Data {data.name} is auto-correlated")
        else:
            logging.info(f"Data {data.name} is not auto-correlated")
        model = train_auto_regressive_model(data)
        return forecast(model, data)

    return predict