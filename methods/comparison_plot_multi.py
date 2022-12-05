"""Plot a comparison of the ground truth and forecast."""

from typing import Callable, TypeVar, Tuple, List

from predictions.Prediction import PredictionData

Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction")
ConfidenceInterval = TypeVar("ConfidenceInterval")
Title = TypeVar("Title", covariant=True)

from .plot import Figure, Plot


def comparison_plot_multi(
    plot_full_dataset_plus_predictions: Callable[[Data, List[PredictionData]], Figure],
    get_plot_metadata: Callable[[List[PredictionData]], Tuple[str, str, str]],
    save_plot: Callable[[Figure, str, str, str], None],
) -> Plot[Data, Prediction, ConfidenceInterval, Title]:
    def draw_plot(
        training_dataset: Data,
        predictions: List[PredictionData],
    ) -> None:
        full_figure = plot_full_dataset_plus_predictions(training_dataset, predictions)
        plot_folder, plot_file_name, title = get_plot_metadata(predictions)
        return save_plot(full_figure, plot_folder, plot_file_name, "full", title)

    return draw_plot
