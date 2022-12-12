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


def __compile_results(list_of_list_of_reports: List[List[Report]]) -> pd.DataFrame:
    """Compile the results from a list of lists of reports into a dataframe"""
    results = []
    for list_of_method_results_per_dataset in list_of_list_of_reports:
        for report in list_of_method_results_per_dataset:
            results.append(
                {
                    "method": report.method,
                    "dataset": report.dataset.name,
                    "subset_row": report.dataset.subset_row_name,
                    "MAE": report.metrics["MAE"],
                    "RMSE": report.metrics["RMSE"],
                    "MAPE": report.metrics["MAPE"],
                }
            )
    results = pd.DataFrame(results)
    return results


def __get_title(results_dataframe: pd.DataFrame, chosen_metric: str) -> str:
    """Get the title of the plot"""
    return f"{chosen_metric} of {results_dataframe['method'].unique().size} methods on {results_dataframe['dataset'].unique().size} datasets"


def __plot_heatmap(
    results_dataframe: pd.DataFrame, chosen_metric: str = "MAE"
) -> Figure:
    """Plot the results in a heatmap"""
    title = __get_title(results_dataframe, chosen_metric)
    figure, axis = plt.subplots(figsize=(20, 10))
    axis.set_title(title)
    axis = sns.heatmap(
        results_dataframe.pivot(columns="method", index="subset_row", values="MAE"),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        ax=axis,
    )
    return figure


def __get_time_stamp_for_file_name() -> str:
    """Get the time stamp for the file name"""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def __save_plot(figure: Figure, folder: str, file_name: str) -> None:
    """Save the plot"""
    logging.info(f"Saving heatmap to plots/{folder}/{file_name}_heat_map.png")
    if not os.path.exists(f"plots/{folder}"):
        os.makedirs(f"plots/{folder}")

    time_stamp_string = __get_time_stamp_for_file_name()
    figure.savefig(
        f"plots/{folder}/{file_name}_heat_map_{time_stamp_string}.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(figure.figure)


plot_results_in_heatmap = method(__compile_results, __plot_heatmap, __save_plot)
