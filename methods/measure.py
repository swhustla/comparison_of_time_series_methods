from typing import Protocol, TypeVar

from .predict import Predict, Prediction

Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction", covariant=True)
Metrics = TypeVar("Metrics", covariant=True)

class Measure(Protocol[Data, Prediction, Metrics]):
    def __call__(self, predict: Predict[Data, Prediction]) -> Metrics:
        pass


