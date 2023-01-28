from typing import Protocol, TypeVar

from .predict import Predict, Prediction


Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction", covariant=True)
ConfidenceInterval = TypeVar("ConfidenceInterval", covariant=True)
Title = TypeVar("Title", covariant=True)
Figure = TypeVar("Figure")

class Plot(Protocol[Data, Prediction, ConfidenceInterval, Title]):
    def __call__(self, predict: Predict[Data, Prediction]) -> None:
        pass
