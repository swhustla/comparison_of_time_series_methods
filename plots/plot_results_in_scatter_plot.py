"""Plot a scatterplot  using seaborn showing the accuracy (for a given metric) 
versus time taken (seconds) of lots of time series prediction methods 
for a single time series"""


import os
import logging
import datetime

import numpy as np

from plots.color_map_by_method import __color_map_by_method_dict

import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import pandas as pd

from methods.plot import Figure
from predictions.Prediction import PredictionData

from data.report import Report

from plots.plot_results_in_heatmap import __get_dataset_name, __get_time_stamp_for_file_name
from methods.plot_results_in_scatter_plot import plot_results_in_scatter_plot as method

from methods.plot_results_in_scatter_plot import plot_results_in_scatter_plot_from_csv as method_report_from_csv


def __compile_results_single_dataset(list_of_reports: List[Report]) -> Tuple[pd.DataFrame, str]:
    """Compile the results from a list of lists of reports into a dataframe"""
    results = []
    
    for report in list_of_reports:
        if report.prediction.number_of_iterations > 1:
            time_elapsed = (report.end_time - report.tstart) / report.prediction.number_of_iterations
        else:
            time_elapsed = report.end_time - report.tstart
        results.append(
            {
                "method": report.method,
                "dataset": report.dataset.name,
                "subset_row": report.dataset.subset_row_name,
                "MAE": report.metrics["mean_absolute_error"],
                "RMSE": report.metrics["root_mean_squared_error"],
                "R2": report.metrics["r_squared"],
                "MAPE": report.metrics["mean_absolute_percentage_error"],
                "Elapsed (s)": np.round(time_elapsed, 2),
            }
        )
    return pd.DataFrame(results)


def __get_title(results_dataframe: pd.DataFrame, chosen_metric: str) -> str:
    """Get the title of the plot"""
    return f"{chosen_metric} results for {__get_dataset_name(results_dataframe)} for {results_dataframe['method'].unique().size} predictive methods on {results_dataframe['subset_row'].unique()[0]} dataset"


def __plot_scatterplot(
    results_dataframe: pd.DataFrame, chosen_metric: str = "MAE"
) -> Figure:
    """Plot the results in a scatterplot, accuracy metric versus elapsed time"""
    logging.info(f"Plotting {chosen_metric} scatterplot")

    title = __get_title(results_dataframe, chosen_metric)

    # use colow map by method dictionary
    color_map = __color_map_by_method_dict

    # plot the results
    sns.set_theme(style="whitegrid")
    sns.set_palette("dark")
    sns.set_color_codes("dark")
    fig, ax = plt.subplots(figsize=(14, 9))
    # set the color of the x and y axis to black
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    sns.scatterplot(
        data=results_dataframe,
        x="Elapsed (s)",
        y=chosen_metric,
        hue="method",
        ax=ax,
        palette=color_map,
        marker="X",
        s=200,
    )
    plt.xscale("log")

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Elapsed (s)", fontsize=18, weight="semibold")
    ax.set_ylabel(chosen_metric, fontsize=18, weight="semibold")
    for spine in ax.spines.values():
        spine.set_linewidth(2)
   
    # Set axis tick parameters
    ax.tick_params(axis="both", which="major", labelsize=22, width=2, labelrotation=50)
    plt.legend(fontsize="xx-large", title_fontsize="20", loc="upper right")
    return fig


def __get_subset_row(results_dataframe: pd.DataFrame) -> str:
    """Get the subset row name"""
    return results_dataframe["subset_row"].unique()[0]


def __get_filename(results_dataframe: pd.DataFrame, chosen_metric: str) -> str:
    """Get the filename for the plot"""
    folder = f"plots/{__get_dataset_name(results_dataframe)}/{__get_subset_row(results_dataframe)}/scatter_plots"
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    return f"{folder}/{chosen_metric}_{__get_time_stamp_for_file_name()}.png"


def __save_plot(figure: Figure, results_dataframe: pd.DataFrame, chosen_metric: str):
    """Save the scatter plot to a file"""
    filename = __get_filename(results_dataframe, chosen_metric)
    logging.info(f"Saving scatter plot to {filename}")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    figure.savefig(filename)

    plt.close(figure.figure)


plot_results_in_scatter_plot = method(__compile_results_single_dataset, __plot_scatterplot, __save_plot)

plot_results_in_scatter_plot_from_csv =  method_report_from_csv( __plot_scatterplot, __save_plot)





