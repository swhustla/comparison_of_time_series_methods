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
import matplotlib.ticker as ticker
from typing import List, Tuple, Optional
import pandas as pd

from methods.plot import Figure
from predictions.Prediction import PredictionData

from data.report import Report

from plots.plot_results_in_heatmap import (
    __get_dataset_name,
    __get_time_stamp_for_file_name,
    __get_plot_params,
)
from methods.plot_results_in_scatter_plot import plot_results_in_scatter_plot as method

from methods.plot_results_in_scatter_plot import (
    plot_results_in_scatter_plot_from_csv as method_report_from_csv,
    plot_results_in_scatter_plot_multi_from_csv as method_report_multi_from_csv,
)


def __compile_results_single_dataset(
    list_of_reports: List[Report],
) -> Tuple[pd.DataFrame, str]:
    """Compile the results from a list of lists of reports into a dataframe"""
    results = []

    for report in list_of_reports:
        if report.prediction.number_of_iterations > 1:
            time_elapsed = (
                report.end_time - report.tstart
            ) / report.prediction.number_of_iterations
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


def __get_title_multi(
    results_dataframe: pd.DataFrame, chosen_metric: str, group_name: str
) -> str:
    """Get the title of the plot"""
    if group_name is None:
        return f"{chosen_metric} results for {__get_dataset_name(results_dataframe)} for {results_dataframe['method'].unique().size} predictive methods on {results_dataframe['subset_row'].unique().size} datasets"
    else:
        return f"{chosen_metric} results for {group_name} with {results_dataframe['method'].unique().size} predictive methods"


def __plot_scatterplot(
    results_dataframe: pd.DataFrame, chosen_metric: str = "MAE"
) -> Figure:
    """Plot the results in a scatterplot, accuracy metric versus elapsed time"""
    logging.info(f"Plotting {chosen_metric} scatterplot")

    title = __get_title(results_dataframe, chosen_metric)

    # use colow map by method dictionary
    color_map = __color_map_by_method_dict
    # plot the results
    fig, ax = plt.subplots(figsize=(15, 11))
    sns.set_theme(
        style="whitegrid",
        rc={
            "xtick.bottom": True,
            "ytick.left": True,
        },
    )
    # Customize the style to keep axis lines and ticks in black
    sns.set_style(
        {"axes.edgecolor": "black", "xtick.color": "black", "ytick.color": "black"}
    )   

    sns.scatterplot(
        data=results_dataframe,
        x="Elapsed (s)",
        y=chosen_metric,
        hue="method",
        ax=ax,
        palette=color_map,
        marker="X",
        s=800,
    )
    # set more ticks on the x-axis
    ax.set_xscale("log")
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    x_ticks = ticker.LogLocator(
        base=10.0, subs=np.arange(1, np.log(x_range), 2), numticks=10
    )
    ax.xaxis.set_major_locator(x_ticks)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # chose the metric to plot: since R2 has negative values we use a symlog scale
    if chosen_metric == "R2":
        ax.set_yscale("symlog")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    else:
        # set more ticks on the y-axis
        ax.set_yscale("log")
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_ticks = ticker.LogLocator(base=10.0, subs=np.arange(0.1, np.log(y_range), 2))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_locator(y_ticks)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Elapsed (s)", fontsize=18, weight="semibold")
    ax.set_ylabel(chosen_metric, fontsize=18, weight="semibold")
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Set axis tick parameters
    ax.tick_params(axis="both", which="major", labelsize=22, width=2, labelrotation=50)
    plt.legend(fontsize="xx-large", title_fontsize="20", loc="upper right")
    return fig


def __plot_scatterplot_multi(
    results_dataframe: pd.DataFrame,
    chosen_metric: str = "MAE",
    group_name: Optional[str] = None,
) -> Figure:
    """Plot the results in a scatterplot, accuracy metric versus elapsed time"""
    logging.info(f"Plotting {chosen_metric} scatter plot multi")

    title = __get_title_multi(results_dataframe, chosen_metric, group_name)

    # use colow map by method dictionary
    color_map = __color_map_by_method_dict
    # plot the results
    fig, ax = plt.subplots(figsize=(15, 11))
    sns.set_theme(
        style="whitegrid",
        rc={
            "xtick.bottom": True,
            "ytick.left": True,
        },
    )
    # Customize the style to keep axis lines and ticks in black
    sns.set_style(
        {"axes.edgecolor": "black", "xtick.color": "black", "ytick.color": "black"}
    )

    # group the dataframe by 'Name' and calculate the mean for each group
    results_dataframe_avg = results_dataframe.groupby("method").mean()

    sns.scatterplot(
        data=results_dataframe,
        x="Elapsed (s)",
        y=chosen_metric,
        hue="method",
        ax=ax,
        palette=color_map,
        marker="X",
        s=200,
        alpha=0.2,
    )

    # set more ticks on the x-axis
    ax.set_xscale("log")
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    x_ticks = ticker.LogLocator(
        base=10.0, subs=np.arange(1, np.log(x_range), 2), numticks=10
    )
    ax.xaxis.set_major_locator(x_ticks)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # chose the metric to plot: since R2 has negative values we use a symlog scale
    if chosen_metric == "R2":
        ax.set_yscale("symlog")
    else:
        # set more ticks on the y-axis
        ax.set_yscale("log")
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        y_ticks = ticker.LogLocator(base=10.0, subs=np.arange(1, np.log(y_range), 2))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_major_locator(y_ticks)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    sns.scatterplot(
        data=results_dataframe_avg,
        x="Elapsed (s)",
        y=chosen_metric,
        hue="method",
        ax=ax,
        palette=color_map,
        marker="X",
        s=800,
        linewidth=1,
        legend=False,
    )

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Elapsed (s)", fontsize=18, weight="semibold")
    ax.set_ylabel(chosen_metric, fontsize=18, weight="semibold")
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Set axis tick parameters
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=22,
        width=2,
        length=4,
        color="black",
        pad=4,
    )
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


def __save_plot_multi(
    figure: Figure,
    dataset_name: str,
    plot_type: str,
    chosen_metric: str,
    file_format="png",
) -> None:
    """Save a scatter plot multi."""
    time_stamp_string = __get_time_stamp_for_file_name()
    folder_location = f"plots/{dataset_name}/{chosen_metric}_scatter_plots"
    os.makedirs(folder_location, exist_ok=True)

    filename = f"scatter_plots_{plot_type}_{time_stamp_string}.{file_format}"
    counter = 1
    while os.path.exists(os.path.join(folder_location, filename)):
        counter += 1
        filename = (
            f"scatter_plots_{plot_type}_{time_stamp_string}_{counter}.{file_format}"
        )

    filepath = os.path.join(folder_location, filename)
    logging.info(f"Saving scatter plots multi to {filepath}")
    figure.savefig(filepath, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure.figure)


plot_results_in_scatter_plot = method(
    __compile_results_single_dataset, __plot_scatterplot, __save_plot
)

plot_results_in_scatter_plot_from_csv = method_report_from_csv(
    __plot_scatterplot, __save_plot
)

plot_results_in_scatter_plot_multi_from_csv_ = method_report_multi_from_csv(
    __plot_scatterplot_multi, __get_plot_params, __save_plot_multi
)
