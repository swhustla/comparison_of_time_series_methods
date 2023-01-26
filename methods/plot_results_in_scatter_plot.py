"""Plot a scatterplot  using seaborn showing the accuracy of lots of time series prediction methods
on lots of different time series"""

from typing import List, Tuple, Optional, Callable, TypeVar, Generator

from data.report import Report
import pandas as pd
from methods.plot import Figure

Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction")
ConfidenceInterval = TypeVar("ConfidenceInterval")
Title = TypeVar("Title", covariant=True)


from .plot import Figure, Plot


__chosen_metrics = ["MAE", "RMSE", "R2", "MAPE"]

def plot_results_in_scatter_plot(
    compile_results_single_dataset: Callable[[List[List[Report]]],Data],
    plot_scatterplot: Callable[[Data, str], Figure],
    save_plot: Callable[[Figure, pd.DataFrame, str], None],
) -> Plot[Data, Prediction, ConfidenceInterval, Title]:
    def draw_plot(
        list_of_list_of_reports: List[List[Report]]
    ) -> None:
        print("Plotting results in scatter plot")
        results_dataframe = compile_results_single_dataset(list_of_list_of_reports)
        for chosen_metric in __chosen_metrics:
            figure = plot_scatterplot(results_dataframe, chosen_metric)
            save_plot(figure, results_dataframe, chosen_metric)

    return draw_plot