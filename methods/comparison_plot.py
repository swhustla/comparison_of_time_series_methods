"""Plot a comparison of the ground truth and forecast."""

from typing import Callable, TypeVar

from predictions.Prediction import PredictionData

Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction")
ConfidenceInterval = TypeVar("ConfidenceInterval")
Title = TypeVar("Title", covariant=True)

from .plot import Figure, Plot

def comparison_plot(
    plot: Callable[[PredictionData], Figure],
    save_plot: Callable[[Figure, Title], None],
) -> Plot[Data, Prediction, ConfidenceInterval, Title]:
    def draw_plot(
        prediction: PredictionData,
    ) -> Figure:
        figure = plot(prediction)
        return save_plot(figure, prediction.title)

    return draw_plot