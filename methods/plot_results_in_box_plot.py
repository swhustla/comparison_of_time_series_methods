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


def plot_results_in_boxplot_from_csv(
    plot_boxplot_by_method: Callable[[Data, str], Figure],
    plot_boxplot_by_data: Callable[[Data, str], Figure],
    get_plot_params: Callable[[Figure, str, Data, str], List],
    save_plot_boxplot: Callable[[Figure, str, str, str], None],
) -> Plot[Data, Prediction, ConfidenceInterval, Title]:
    def draw_plot(data_to_plot: Data, dataset_name: str) -> None:
        print("Plotting results in boxplot; source: csv")
        for chosen_metric in __chosen_metrics:
            figure_method = plot_boxplot_by_method(data_to_plot, chosen_metric)
            figure_data = plot_boxplot_by_data(data_to_plot, chosen_metric)
            plot_params = get_plot_params(
                figure_method, chosen_metric, data_to_plot, dataset_name
            )
            print(f"Saving {dataset_name} {chosen_metric} boxplot plot")
            #save boxplot ploted over methods
            save_plot_boxplot(*plot_params[:-1]),
            #save boxplot ploted over datasets
            save_plot_boxplot(
                figure_data, plot_params[1], plot_params[4], chosen_metric
            ),

    return draw_plot
