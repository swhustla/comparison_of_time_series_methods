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
    save_plot: Callable[[Figure, str, str, str], None],
) -> Plot[Data, Prediction, ConfidenceInterval, Title]:
    def draw_plot(
        list_of_list_of_reports: List[List[Report]],
        group_name: str,
    ) -> None:
        print("Plotting results in heatmap: source: report")
        results_dataframe, dataset_name = compile_results(list_of_list_of_reports)
        for chosen_metric in __chosen_metrics:
            figure = plot_heatmap(results_dataframe, chosen_metric, group_name)
            print(f"Saving {dataset_name} {chosen_metric} heatmap plot")
            save_plot(figure, dataset_name, chosen_metric)

    return draw_plot


def plot_results_in_heatmap_from_csv(
    plot_heatmap: Callable[[Data, str], Figure],
    save_plot: Callable[[Figure, str, str, str], None],
) -> Plot[Data, Prediction, ConfidenceInterval, Title]:
    def draw_plot(data_to_plot: Data, dataset_name: str) -> None:
        print("Plotting results in heatmap; source: csv")
        for chosen_metric in __chosen_metrics:
            figure = plot_heatmap(data_to_plot, chosen_metric)
            stock_company_name = data_to_plot["subset_row"][0]
            print(f"Saving {dataset_name} {chosen_metric} heatmap plot")
            input_map = {
                ("Stock price", "HD"): (figure, dataset_name, "young", chosen_metric),
                ("Stock price", "GOOG"): (figure, dataset_name, "old", chosen_metric),
                ("India city pollution", "Ahmedabad"): (
                    figure,
                    "Indian city pollution",
                    "_",
                    chosen_metric,
                ),
            }
            input_key = (dataset_name, stock_company_name)

            if input_key in input_map:
                plot_params = input_map[input_key]
                save_plot(*plot_params)

    return draw_plot