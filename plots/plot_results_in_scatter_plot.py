"""Plot a scatterplot  using seaborn showing the accuracy (for a given metric) 
versus time taken (seconds) of lots of time series prediction methods 
for a single time series"""


import os
import logging
import datetime

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import pandas as pd

from methods.plot import Figure
from predictions.Prediction import PredictionData

from data.report import Report

from plots.plot_results_in_heatmap import __get_dataset_name, __get_time_stamp_for_file_name
from methods.plot_results_in_scatter_plot import plot_results_in_scatter_plot as method



def __compile_results_single_dataset(list_of_reports: List[Report]) -> Tuple[pd.DataFrame, str]:
    """Compile the results from a list of lists of reports into a dataframe"""
    results = []
    for report in list_of_reports:
        results.append(
            {
                "method": report.method,
                "dataset": report.dataset.name,
                "subset_row": report.dataset.subset_row_name,
                "MAE": report.metrics["mean_absolute_error"],
                "RMSE": report.metrics["root_mean_squared_error"],
                "R2": report.metrics["r_squared"],
                "MAPE": report.metrics["mean_absolute_percentage_error"],
                "Elapsed (s)": report["Elapsed (s)"],
            }
        )
    results = pd.DataFrame(results)
    dataset_name = list_of_reports[0].dataset.name
    return results, dataset_name


def __get_title(results_dataframe: pd.DataFrame, chosen_metric: str) -> str:
    """Get the title of the plot"""
    return f"{chosen_metric} results for {__get_dataset_name(results_dataframe)} for {results_dataframe['method'].unique().size} predictive methods on {results_dataframe['subset_row'].unique()[0]} dataset"


def __plot_scatterplot(
    results_dataframe: pd.DataFrame, chosen_metric: str = "MAE"
) -> Figure:
    """Plot the results in a scatterplot, accuracy metric versus elapsed time"""
    logging.info(f"Plotting {chosen_metric} scatterplot")

    title = __get_title(results_dataframe, chosen_metric)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.scatterplot(
        data=results_dataframe,
        x="Elapsed (s)",
        y=chosen_metric,
        hue="method",
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Elapsed (s)")
    ax.set_ylabel(chosen_metric)
    return Figure(fig, title)


def __get_filename(results_dataframe: pd.DataFrame, chosen_metric: str) -> str:
    """Get the filename for the plot"""
    return f"plots/{__get_dataset_name(results_dataframe)}/{__get_time_stamp_for_file_name(results_dataframe)}_{chosen_metric}_scatterplot.png"


def __save_plot(figure: Figure, results_dataframe: pd.DataFrame, chosen_metric: str):
    """Save the scatter plot to a file"""
    filename = __get_filename(results_dataframe, chosen_metric)
    logging.info(f"Saving scatter plot to {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    figure.save(filename)

    plt.close(figure.fig)


plot_results_in_scatter_plot = method(__compile_results_single_dataset, __plot_scatterplot, __save_plot)






