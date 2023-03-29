"""Plot a heatmap plot showing the accuracy of lots of time series prediction methods 
on lots of different time series"""
import os
import logging
import datetime

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable, TypeVar, Generator
import pandas as pd

Figure = TypeVar("Figure")
from predictions.Prediction import PredictionData

from data.report import Report
from data.stock_prices import (
    get_a_list_of_growth_stock_tickers,
    get_a_list_of_value_stock_tickers,
)


__dataset_group_titles: dict[str, str] = {
    "Indian city pollution": "Cities on the Indo-Gangetic Plain in India",
    "stock_prices": "Value stocks",
}

from methods.plot_results_in_heatmap import plot_results_in_heatmap as method_report
from methods.plot_results_in_heatmap import (
    plot_results_in_heatmap_from_csv as method_report_from_csv,
)


def __compile_results(
    list_of_list_of_reports: List[List[Report]],
) -> Tuple[pd.DataFrame, str]:
    """Compile the results from a list of lists of reports into a dataframe"""
    results = []
    for list_of_method_results_per_dataset in list_of_list_of_reports:
        for report in list_of_method_results_per_dataset:
            results.append(
                {
                    "method": report.method,
                    "dataset": report.dataset.name,
                    "subset_row": report.dataset.subset_row_name,
                    "MAE": report.metrics["mean_absolute_error"],
                    "RMSE": report.metrics["root_mean_squared_error"],
                    "R2": report.metrics["r_squared"],
                    "MAPE": report.metrics["mean_absolute_percentage_error"],
                }
            )
    results = pd.DataFrame(results)
    dataset_name = list_of_list_of_reports[0][0].dataset.name
    return results, dataset_name


def __get_dataset_name(results_dataframe: pd.DataFrame) -> str:
    """Get the name of the dataset"""
    if results_dataframe["dataset"].unique()[0] in __dataset_group_titles:
        return __dataset_group_titles[results_dataframe["dataset"].unique()[0]]
    else:
        return results_dataframe["dataset"].unique()[0]


def __get_title(
    results_dataframe: pd.DataFrame, chosen_metric: str, group_name: str
) -> str:
    """Get the title of the plot"""
    if group_name is None:
        return f"{chosen_metric} results for {__get_dataset_name(results_dataframe)} for {results_dataframe['method'].unique().size} predictive methods on {results_dataframe['subset_row'].unique().size} datasets"
    else:
        return f"{chosen_metric} results for {group_name} with {results_dataframe['method'].unique().size} predictive methods"


def __plot_heatmap(
    results_dataframe: pd.DataFrame,
    chosen_metric: str = "MAE",
    group_name: Optional[str] = None,
) -> Figure:
    """Plot the results in a heatmap"""
    logging.info(f"Plotting {chosen_metric} heatmap")

    title = __get_title(results_dataframe, chosen_metric, group_name)
    figure, axis = plt.subplots(figsize=(20, 10))
    axis.set_title(title)
    axis.set_xlabel("Method")
    if group_name is None:
        axis.set_ylabel("Dataset")
    else:
        axis.set_ylabel(f"Dataset ({group_name})")
    # chose sns colormap that goes from red (high error) to green (low error) without white in the middle
    colormap = sns.diverging_palette(220, 20, as_cmap=True)

    # condensed distance matrix must contain only finite values
    results_dataframe = results_dataframe.copy()
    results_dataframe.loc[:, chosen_metric] = results_dataframe[chosen_metric].replace(
        ["none", "nan"], np.nan
    )

    pivoted_dataframe = results_dataframe.pivot(
        columns="method", index="subset_row", values=chosen_metric
    )
    # condensed distance matrix must contain only finite values
    pivoted_dataframe = pivoted_dataframe.replace([np.inf, -np.inf], np.nan)

    # deal with the case where there are no results for a method
    pivoted_dataframe = pivoted_dataframe.fillna(0)
    mask_for_missing_values = pivoted_dataframe == 0

    # colorbar for MAPE case
    if chosen_metric == "MAPE":
        cluster_grid = sns.clustermap(
            pivoted_dataframe,
            annot=True,
            fmt=".2f",
            cmap="rainbow",
            vmin=pivoted_dataframe.min().min(),
            vmax=60,
            mask=mask_for_missing_values,
        )
        cluster_grid.ax_col_dendrogram.set_title(title)
        return cluster_grid

    # reversing the colorbar for R2 case
    if chosen_metric == "R2":
        cluster_grid = sns.clustermap(
            pivoted_dataframe,
            annot=True,
            fmt=".2f",
            cmap=colormap.reversed(),
            vmin=-1,
            vmax=1,
            mask=mask_for_missing_values,
        )
        cluster_grid.ax_col_dendrogram.set_title(title)
        return cluster_grid

    cluster_grid = sns.clustermap(
        pivoted_dataframe,
        annot=True,
        fmt=".2f",
        cmap=colormap,
        vmin=0,
        vmax=100,
        mask=mask_for_missing_values,
    )
    cluster_grid.ax_col_dendrogram.set_title(title)
    return cluster_grid


def __get_time_stamp_for_file_name() -> str:
    """Get the time stamp for the file name"""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def __get_plot_params(
    figure: Figure, metric: str, data_to_plot: pd.DataFrame, dataset_name: str
):
    """Get the parameters for the plot"""
    sub_category_name = data_to_plot["subset_row"][0]
    input_map = {}
    if dataset_name == "Stock price":
        list_of_growth = get_a_list_of_growth_stock_tickers()
        for ticker in list_of_growth:
            input_map[(dataset_name, ticker)] = (figure, dataset_name, "growth", metric)
        list_of_value = get_a_list_of_value_stock_tickers()
        for ticker in list_of_value:
            input_map[(dataset_name, ticker)] = (figure, dataset_name, "value", metric)
    elif dataset_name in ["India city pollution", "Indian city pollution"]:
        input_map[(dataset_name, sub_category_name)] = (
            figure,
            "Indian city pollution",
            "_",
            metric,
        )
    else:
        raise ValueError("Invalid dataset_name: {}".format(dataset_name))
    input_key = (dataset_name, sub_category_name)
    input_value = input_map.get(input_key, "__")

    return input_value


def __save_heatmap(
    figure: Figure,
    dataset_name: str,
    plot_type: str,
    chosen_metric: str,
    file_format="png",
) -> None:
    """Save a heatmap plot."""
    time_stamp_string = __get_time_stamp_for_file_name()
    folder_location = f"plots/{dataset_name}/{chosen_metric}_heatmaps"
    os.makedirs(folder_location, exist_ok=True)

    filename = f"heat_map_{plot_type}_{time_stamp_string}.{file_format}"
    counter = 1
    while os.path.exists(os.path.join(folder_location, filename)):
        counter += 1
        filename = f"heat_map_{plot_type}_{time_stamp_string}_{counter}.{file_format}"

    filepath = os.path.join(folder_location, filename)
    logging.info(f"Saving heatmap to {filepath}")
    figure.savefig(filepath, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure.figure)


plot_results_in_heatmap = method_report(
    __compile_results, __plot_heatmap, __get_plot_params, __save_heatmap
)
plot_results_in_heatmap_from_csv = method_report_from_csv(
    __plot_heatmap, __get_plot_params, __save_heatmap
)
