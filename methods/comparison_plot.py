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
    save_plot: Callable[[Figure, str, str, str], None],
) -> Plot[Data, Prediction, ConfidenceInterval, Title]:
    def draw_plot(
        training_data: Data,
        prediction: PredictionData,
    ) -> Tuple[None, None]:
        full_figure = full_data_plus_prediction_plot(training_data, prediction)
        figure = plot(prediction)
        return (
            save_plot(
                figure, prediction.plot_folder, prediction.plot_file_name, "comparison", prediction.title
            ),
            save_plot(full_figure, prediction.plot_folder, prediction.plot_file_name, "full", prediction.title),
        )

    return draw_plot
