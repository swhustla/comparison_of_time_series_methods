"""
Tsetlin Machine Regression method

This module contains the Tsetlin Machine Regression method.
It is a wrapper around the pyTsetlinMachine library.

"""

from typing import Tuple, TypeVar, Callable, Dict
from data.dataset import Dataset
from predictions.Prediction import PredictionData
import logging

from .predict import Predict

Data = TypeVar("Data", contravariant=True)
Model = TypeVar("Model")
Prediction = TypeVar("Prediction")

def tsetlin_machine(
    split_data: Callable[[Dataset], Tuple[Data, Data, Data, Data]],
    create_tsetlin_machine_regression_model: Callable[[], Model],
    train_tsetlin_machine_regression_model: Callable[[Model, Data, Data], Model],
    predict_tsetlin_machine_regression_model: Callable[[Model, Data], Prediction],
) -> Predict[Dataset, PredictionData]:
    """
    Return a function that takes a dataset and returns a prediction.
    """
    def predict(dataset: Dataset) -> PredictionData:
        """
        Return a prediction for the given dataset.
        """
        logging.info("Tsetlin Machine Regression method")
        train_data, test_data, train_labels, test_labels = split_data(dataset)
        model = create_tsetlin_machine_regression_model()
        model = train_tsetlin_machine_regression_model(model, train_data, train_labels)
        prediction = predict_tsetlin_machine_regression_model(model, test_data)
        return PredictionData(prediction, test_labels)
    return predict




