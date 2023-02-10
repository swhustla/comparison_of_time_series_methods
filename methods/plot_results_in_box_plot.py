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
    plot_boxplot_by_city: Callable[[Data, str], Figure],
    save_plot_boxplot: Callable[[Figure, str, str, str], None],
) -> Plot[Data, Prediction, ConfidenceInterval, Title]:
    def draw_plot(
        data_to_plot: Data,
        dataset_name: str
    ) -> None:
        print("Plotting results in boxplot; source: csv")
        for chosen_metric in __chosen_metrics:
            figure_method = plot_boxplot_by_method(data_to_plot, chosen_metric)
            figure_city = plot_boxplot_by_city(data_to_plot, chosen_metric)
            stock_company_name = data_to_plot["subset_row"][0]
            print(f"Saving {dataset_name} {chosen_metric} boxplot plot")
        if dataset_name =='Stock price' and stock_company_name == 'HD':
            return (
                save_plot_boxplot(
                    figure_method, dataset_name,'by_method_young', chosen_metric
                ),
                save_plot_boxplot(figure_city, dataset_name, 'by_data_young', chosen_metric),
            )           
                #save_plot_boxplot(figure_method, figure_city, dataset_name, chosen_metric)
        if dataset_name =='Stock price' and stock_company_name == 'GOOG':
            return (
                save_plot_boxplot(
                    figure_method, dataset_name,'by_method_old', chosen_metric
                ),
                save_plot_boxplot(figure_city, dataset_name, 'by_data_old', chosen_metric),
            )           
                #save_plot_boxplot(figure_method, figure_city, dataset_name, chosen_metric)
        else:
             return (
                save_plot_boxplot(
                    figure_method, dataset_name,'by_method', chosen_metric
                ),
                save_plot_boxplot(figure_city, dataset_name, 'by_data', chosen_metric),
            )           
                #save_plot_boxplot(figure_method, figure_city, dataset_name, chosen_metric)           

    return draw_plot