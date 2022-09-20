"""
FCNN embedding prediction

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


def fcnn_embedding(
    add_features: Callable[[Dataset], Dataset],
    split_data: Callable[[Dataset], tuple[Data, Data, Data, Data]],
    create_model: Callable[[Data, Dataset], Model],
    learn_model: Callable[[Model, Data, Data], Model],
    get_predictions: Callable[[Model, Data], PredictionData],
) -> Predict[Dataset, PredictionData]:
    """Create a Fully Connected Neural Network prediction method."""

    def predict(data: Dataset) -> PredictionData:
        """Predict the given data."""
        
        data = add_features(data)
        x_train, x_test, y_train, y_test = split_data(data)
        model = create_model(x_train, data)
        trained_model = learn_model(model, x_train, y_train)
        return get_predictions(trained_model, data, x_test, y_test)

    return predict
