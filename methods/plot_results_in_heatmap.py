"""Plot a heatmap plot showing the accuracy of lots of time series prediction methods 
on lots of different time series"""

from typing import List, Tuple, Optional, Callable, TypeVar, Generator

from data.report import Report

Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction")
ConfidenceInterval = TypeVar("ConfidenceInterval")
Title = TypeVar("Title", covariant=True)
Figure = TypeVar("Figure")


from .plot import Plot


__chosen_metrics = ["MAE", "RMSE", "R2", "MAPE"]


def plot_results_in_heatmap(
    compile_results: Callable[[List[List[Report]], str], Tuple[Data, str]],
    plot_heatmap: Callable[[Data, str], Figure],
    get_plot_params: Callable[[Figure, str, Data, str], List],
    save_heatmap: Callable[[Figure, str, str, str], None],
) -> Plot[Data, Prediction, ConfidenceInterval, Title]:
    def draw_plot(
        list_of_list_of_reports: List[List[Report]],
        group_name: str,
    ) -> None:
        print("Plotting results in heatmap: source: live report")
        results_dataframe, dataset_name = compile_results(list_of_list_of_reports)
        for chosen_metric in __chosen_metrics:
            figure = plot_heatmap(results_dataframe, chosen_metric, group_name)
            plot_params = get_plot_params(
                figure, chosen_metric, results_dataframe, dataset_name
            )
            print(f"Saving {dataset_name} {chosen_metric} heatmap plot")
            save_heatmap(*plot_params)

    return draw_plot


def plot_results_in_heatmap_from_csv(
    plot_heatmap: Callable[[Data, str], Figure],
    get_plot_params: Callable[[Figure, str, Data, str], List],
    save_heatmap: Callable[[Figure, str, str, str], None],
) -> Plot[Data, Prediction, ConfidenceInterval, Title]:
    def draw_plot(data_to_plot: Data, dataset_name: str) -> None:
        print("Plotting results in heatmap; source: stored csv")
        for chosen_metric in __chosen_metrics:
            print(chosen_metric)
            figure = plot_heatmap(data_to_plot, chosen_metric)
            plot_params = get_plot_params(
                figure, chosen_metric, data_to_plot, dataset_name
            )
            print(f"Saving {dataset_name} {chosen_metric} heatmap plot")
            save_heatmap(*plot_params)

    return draw_plot
