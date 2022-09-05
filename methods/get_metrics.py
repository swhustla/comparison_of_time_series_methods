"""Get all error metrics for a given prediction and ground truth."""

from typing import TypeVar, Callable, Tuple, Dict

from methods.linear_regression import Prediction

Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction")
Metrics = TypeVar("Metrics", covariant=True)

from .measure import Measure


def get_metrics(
    metrics: Callable[[Data, Prediction], Dict[str, float]],
) -> Measure[Data, Prediction, Metrics]:
    def measure(
        ground_truth: Data,
        prediction: Data,
    ) -> Metrics:
        return metrics(ground_truth, prediction)

    return measure
