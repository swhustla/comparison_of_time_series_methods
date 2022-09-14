"""
FCNN prediction

This module contains the Fully Connected Neural Network Prediction method.
It is a wrapper around the keras library.

"""

from typing import TypeVar, Callable
from data.Data import Dataset
from predictions.Prediction import PredictionData

Data = TypeVar("Data", contravariant=True)
Model = TypeVar("Model")
Prediction = TypeVar("Prediction")

from .predict import Predict


def fcnn(
    create_model: Callable[[Dataset], Model],
    split_data: Callable[[Dataset], tuple[Data, Data, Data, Data, Data, Data]],
    learn_model: Callable[[Model, Data, Data], Model],
    get_predictions: Callable[[Model, Data], PredictionData],
) -> Predict[Dataset, PredictionData]:
    """Create a Fully Connected Neural Network prediction method."""

    def predict(data: Dataset) -> PredictionData:
        """Predict the given data."""
        model = create_model(data)
        x_train, x_test, y_train, _, _, y_val = split_data(data)
        model = learn_model(model, x_train, y_train)
        return get_predictions(model, data, x_test, y_val)

    return predict
