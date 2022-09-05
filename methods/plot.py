from typing import Protocol, TypeVar

from .predict import Predict, Prediction


Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction", covariant=True)
Figure = TypeVar("Figure", covariant=True)

class Plot(Protocol[Data, Prediction, Figure]):
    def __call__(self, predict: Predict[Data, Prediction]) -> Figure:
        pass
