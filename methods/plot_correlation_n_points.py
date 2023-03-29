"""Plot a correlation plot showing the correlation between the number of data points in the training set and the metric
 chosen. The metric chosen can be MAE, RMSE, R2 or MAPE. The plot is saved in the folder plots/correlation_Npoints_vs_metric.
 """

from typing import List, Tuple, Optional, Callable, TypeVar, Generator

from data.report import Report
from matplotlib import pyplot as plt

Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction")
ConfidenceInterval = TypeVar("ConfidenceInterval")
Title = TypeVar("Title", covariant=True)
Figure = TypeVar("Figure")

from .plot import Plot

__chosen_metrics = ["MAE", "RMSE", "R2", "MAPE"]


def plot_correlation_Npoints_vs_metric(
    plot_correlation: Callable[[Data, str], Figure],
    save_plot: Callable[[Figure, str, str, str], None],
) -> Plot[Data, Prediction, ConfidenceInterval, Title]:
    def draw_plot(
        training_dataset: List[Data],
        results_dataframe: Data,
    ) -> None:
        print("Plotting results in correlation; source: csv")
        dataset_name = results_dataframe["dataset"].unique()[0]
        for chosen_metric in __chosen_metrics:
            figure = plot_correlation(
                training_dataset, results_dataframe, chosen_metric
            )
            print(f"Saving {dataset_name} {chosen_metric} correlation plot")
            save_plot(figure, results_dataframe, "__", chosen_metric)

    return draw_plot
