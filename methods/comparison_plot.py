"""Plot a comparison of the ground truth and forecast."""

from typing import Callable, TypeVar, Tuple

from predictions.Prediction import PredictionData

Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction")
ConfidenceInterval = TypeVar("ConfidenceInterval")
Title = TypeVar("Title", covariant=True)

from .plot import Figure, Plot

def comparison_plot(
    full_data_plus_prediction_plot: Callable[[Data, PredictionData], Figure],
    plot: Callable[[PredictionData], Figure],
    save_plot: Callable[[Figure, Title, str], None],
) -> Plot[Data, Prediction, ConfidenceInterval, Title]:
    def draw_plot(
        training_data: Data,
        prediction: PredictionData,
    ) -> None:
        full_figure = full_data_plus_prediction_plot(training_data, prediction)
        figure = plot(prediction)
        return (save_plot(figure, prediction.title, "comparison"), save_plot(full_figure, prediction.title, "full"))

    return draw_plot