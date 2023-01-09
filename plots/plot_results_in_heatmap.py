"""Plot a heatmap plot showing the accuracy of lots of time series prediction methods 
on lots of different time series"""
import os
import logging
import datetime

import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import pandas as pd

from methods.plot import Figure
from predictions.Prediction import PredictionData

from data.report import Report


from methods.plot_results_in_heatmap import plot_results_in_heatmap as method


def __compile_results(list_of_list_of_reports: List[List[Report]]) -> Tuple[pd.DataFrame, str]:
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
    return results_dataframe["dataset"].unique()[0]


def __get_title(results_dataframe: pd.DataFrame, chosen_metric: str) -> str:
    """Get the title of the plot"""
    return f"{chosen_metric} results for {__get_dataset_name(results_dataframe)} for {results_dataframe['method'].unique().size} predictive methods on {results_dataframe['subset_row'].unique().size} datasets"


def __plot_heatmap(
    results_dataframe: pd.DataFrame, chosen_metric: str = "MAE"
) -> Figure:
    """Plot the results in a heatmap"""
    logging.info(f"Plotting {chosen_metric} heatmap")

    title = __get_title(results_dataframe, chosen_metric)
    figure, axis = plt.subplots(figsize=(20, 10))
    axis.set_title(title)
    axis.set_xlabel("Method")
    axis.set_ylabel("Dataset")
    # chose sns colormap that goes from red (high error) to green (low error) without white in the middle
    colormap = sns.diverging_palette(220, 20, as_cmap=True)

    #reversing the colorbar for R2 case
    if chosen_metric == 'R2':
        axis = sns.heatmap(
            results_dataframe.pivot(columns="method", index="subset_row", values=chosen_metric),
            annot=True,
            fmt=".2f",
            cmap=colormap.reversed(),
            ax=axis,
        )
        return figure   

    axis = sns.heatmap(
        results_dataframe.pivot(columns="method", index="subset_row", values=chosen_metric),
        annot=True,
        fmt=".2f",
        cmap=colormap,
        ax=axis,
    )
    return figure


def __get_time_stamp_for_file_name() -> str:
    """Get the time stamp for the file name"""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



def __save_plot(figure: Figure, dataset_name: str, chosen_metric: str) -> None:
    """Save the plot"""
    time_stamp_string = __get_time_stamp_for_file_name()
    folder_location = f"plots/{dataset_name}/{chosen_metric}_heatmaps"
    logging.info(f"Saving heatmap to {folder_location}/heat_map_{time_stamp_string}.png")
    if not os.path.exists(f"{folder_location}"):
        os.makedirs(f"{folder_location}")
    
    figure.savefig(
        f"{folder_location}/heat_map_{time_stamp_string}.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(figure.figure)


plot_results_in_heatmap = method(__compile_results, __plot_heatmap, __save_plot)
