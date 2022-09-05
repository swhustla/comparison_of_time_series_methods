"""Plot a comparison of the ground truth and forecast."""

from typing import Callable, TypeVar

Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction")
ConfidenceInterval = TypeVar("ConfidenceInterval")
Title = TypeVar("Title", covariant=True)

from .plot import Figure, Plot

def comparison_plot(
    plot: Callable[[Data, Prediction, ConfidenceInterval, Title], Figure],
    save_plot: Callable[[Figure, Title], None],
) -> Plot[Data, Prediction, ConfidenceInterval, Title]:
    def draw_plot(
        ground_truth: Data,
        prediction: Data,
        confidence_interval: ConfidenceInterval,
        title: Title,
    ) -> Figure:
        figure = plot(ground_truth, prediction, confidence_interval, title)
        return save_plot(figure, title)

    return draw_plot