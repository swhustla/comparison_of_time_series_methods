"""Get all error metrics for a given prediction and ground truth."""

from typing import TypeVar, Callable, Tuple, Dict

from methods.linear_regression import Prediction

from predictions.Prediction import PredictionData

Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction")
Metrics = TypeVar("Metrics", covariant=True)

from .measure import Measure


def get_metrics(
    metrics: Callable[[PredictionData], Dict[str, float]],
) -> Measure[Data, Prediction, Metrics]:
    def measure(
        prediction: PredictionData,
    ) -> Metrics:
        return metrics(prediction)

    return measure
