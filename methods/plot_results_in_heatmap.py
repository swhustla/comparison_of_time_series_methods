"""Plot a heatmap plot showing the accuracy of lots of time series prediction methods 
on lots of different time series"""

from typing import List, Tuple, Optional, Callable, TypeVar, Generator

from data.report import Report
from methods.plot import Figure

Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction")
ConfidenceInterval = TypeVar("ConfidenceInterval")
Title = TypeVar("Title", covariant=True)


from .plot import Figure, Plot


__chosen_metrics = ["MAE", "RMSE", "R2"]

def plot_results_in_heatmap(
    compile_results: Callable[[List[List[Report]]],Tuple[Data, str]],
    plot_heatmap: Callable[[Data, str], Figure],
    save_plot: Callable[[Figure, str, str, str], None],
) -> Plot[Data, Prediction, ConfidenceInterval, Title]:
    def draw_plot(
        list_of_list_of_reports: List[List[Report]]
    ) -> None:
        print("Plotting results in heatmap")
        results_dataframe, dataset_name = compile_results(list_of_list_of_reports)
        for chosen_metric in __chosen_metrics:
            figure = plot_heatmap(results_dataframe, chosen_metric)
            print(f"Saving {dataset_name} {chosen_metric} heatmap plot")
            save_plot(figure, dataset_name, chosen_metric)

    return draw_plot