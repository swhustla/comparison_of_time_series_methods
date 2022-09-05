""" Linear regression 
    A simple machine learning algorithm that 
    fits a linear model to the data."""

from typing import TypeVar, Callable, Tuple

Data = TypeVar("Data", contravariant=True)
Model = TypeVar("Model")
Prediction = TypeVar("Prediction")


from .predict import Predict

def linear_regression(
    train: Callable[[Data], Model],
    test: Callable[[Data, Model], Tuple[Prediction, Data]],
    ) -> Predict[Data, Prediction]:
    def predict(
        data: Data,
    ) -> Prediction:
        trained_model = train(data)
        return test(data, trained_model)
    return predict
    