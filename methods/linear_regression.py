""" Linear regression 
    A simple machine learning algorithm that 
    fits a linear model to the data."""

from typing import TypeVar, Callable

Data = TypeVar("Data", contravariant=True)
Model = TypeVar("Model")
Prediction = TypeVar("Prediction")


from .predict import Predict

def linear_regression(
    train: Callable[[Data], Model],
    test: Callable[[Model, Data], Prediction],
    ) -> Predict[Data, Prediction]:
    def predict(
        data: Data,
    ) -> Prediction:
        trained_model, _ = train(data)
        return test(trained_model, data)
    return predict