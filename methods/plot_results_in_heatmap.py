"""Plot a heatmap plot showing the accuracy of lots of time series prediction methods 
on lots of different time series"""

from typing import List, Tuple, Optional, Callable, TypeVar

from data.report import Report
from methods.plot import Figure

Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction")
Title = TypeVar("Title", covariant=True)


from .plot import Figure, Plot


def plot_results_in_heatmap(
    compile_results: Callable[[List[List[Report]]],Data],
    plot_heatmap: Callable[[Data, str], Figure],
    save_plot: Callable[[Figure, str, str, str], None],
) -> Plot[Data, Prediction, Title]:
    def draw_plot(
        list_of_list_of_reports: List[List[Report]],
        title: str,
        folder: str,
        file_name: str,
    ) -> None:
        results_dataframe = compile_results(list_of_list_of_reports)
        figure = plot_heatmap(results_dataframe, title)
        return save_plot(figure, folder, file_name, "full", title)

    return draw_plot