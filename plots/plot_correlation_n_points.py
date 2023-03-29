"""Plot a correlation plot showing the correlation between the number of data points in the training set and the metric chosen. The metric chosen can be MAE, RMSE, R2 or MAPE. The plot is saved in the folder plots/correlation_Npoints_vs_metric."""
import os
import logging
import datetime

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
from typing import List, Tuple, Optional, Callable, TypeVar, Generator, Union
import pandas as pd

Figure = TypeVar("Figure")
from predictions.Prediction import PredictionData
from plots.color_map_by_method import __color_map_by_method_dict
from methods.plot_correlation_n_points import (
    plot_correlation_Npoints_vs_metric as method_report,
)

from data.report import Report


def __get_dataset_name(results_dataframe: pd.DataFrame) -> str:
    """Get the name of the dataset"""
    return results_dataframe["dataset"].unique()[0]


def __get_title(
    results_dataframe: pd.DataFrame, chosen_metric: str, group_name: str
) -> str:
    """Get the title of the plot"""
    if group_name is None:
        return f"Correlation plot {chosen_metric} for {__get_dataset_name(results_dataframe)} for min/max of {results_dataframe['method'].unique().size} predictive methods on {results_dataframe['subset_row'].unique().size} datasets"
    else:
        return f"Correlation plot {chosen_metric} for {group_name} for min/max of {results_dataframe['method'].unique().size} predictive methods"


def __plot_correlation_npoints_vs_metric(
    training_data: List[pd.DataFrame],
    results_dataframe: pd.DataFrame,
    chosen_metric: str = "MAE",
    group_name: Optional[str] = None,
    plot_all_data: bool = False,  # optional argument
) -> Figure:
    """Plot correlation N data points in training set vs metric."""
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.set_title(__get_title(results_dataframe, chosen_metric, group_name))
     # Replace infinite values with NaN
    results_dataframe[chosen_metric] = results_dataframe[chosen_metric].replace(
        [np.inf, -np.inf], np.nan
    )
     # condensed distance matrix must contain only finite values
    pivoted_dataframe = results_dataframe.pivot(
        columns="method", index="subset_row", values=chosen_metric
    ).reset_index()
    pivoted_dataframe = pivoted_dataframe.replace([np.inf, -np.inf], np.nan)

    # deal with the case where there are no results for a method
    pivoted_dataframe = pivoted_dataframe.fillna(0)

      # Calculate data length using list comprehension
    data_length = [len(data) for data in training_data]
    pivoted_dataframe_Ndata = pivoted_dataframe.copy()
    pivoted_dataframe_Ndata["Ndata"] = data_length
    sns.set_theme(style="whitegrid",rc={
    'xtick.bottom': True,
    'ytick.left': True,
})
    #Customize the style to keep axis lines and ticks in black
    sns.set_style({'axes.edgecolor': 'black', 'xtick.color': 'black', 'ytick.color': 'black'})

    # Create a scatterplot based on the chosen metric
    if chosen_metric == "R2":
        x_data = pivoted_dataframe.min(axis=1, numeric_only=True)
    else:
        x_data = pivoted_dataframe.max(axis=1, numeric_only=True)
    axis = sns.scatterplot(
        data=pivoted_dataframe,
        x=x_data,
        y=data_length,
        hue=pivoted_dataframe["subset_row"],
        palette=sns.color_palette("hls", len(data_length)),
        marker="X",
        s=200,
    )

    # Add labels to each point
    for index, row in pivoted_dataframe.iterrows():
        if chosen_metric == "R2":
            x = pd.to_numeric(row, errors="coerce").min()
        else:
            x = pd.to_numeric(row, errors="coerce").max()
        y = data_length[index]
        name = row["subset_row"]
        bbox_props = dict(
            boxstyle="round,pad=0.5", fc="white", ec="gray", lw=1, alpha=0.5
        )
        axis.text(x, y, name, ha="center", va="center", bbox=bbox_props)

    # Add legend
    axis.legend(fontsize="x-small", title_fontsize="18", loc="upper right")
    
    # Add horizontal and vertical lines
    axis.axhline(y=100, color="gray", linestyle="--")
    axis.axvline(x=-10, color="gray", linestyle="--")

    # Set x-axis range and ticks
    xmin = x_data.min()
    xmax = x_data.max()
    xtick_increment = 0.1
    xticks = [xmin + i * xtick_increment for i in range(int((xmax - xmin) / xtick_increment) + 1)]
    axis.set_xlim(1.1 * xmin, 0.9)
    axis.set_xticks(xticks)
    # Set x-axis scale
    axis.set_xscale("symlog")

    # Set axis tick parameters
    axis.tick_params(axis="both", which="major", labelsize=17, width=2, labelrotation=50)
    axis.xaxis.set_major_formatter(ticker.ScalarFormatter())
    # plot all data points of each method if plot_all_data is True
    if plot_all_data:
        method_vars = [
            "AR",
            "ARIMA",
            "HoltWinters",
            "MA",
            "Prophet",
            "SARIMA",
            "SES",
            "linear_regression",
        ]
        data = pivoted_dataframe_Ndata.melt(
            value_vars=method_vars,
            id_vars="Ndata",
            var_name="series"
        )
        axis = sns.scatterplot(
            x="value",
            hue="series",
            y="Ndata",
            data=data,
        )

    axis.set_xlabel(chosen_metric, fontsize=18, weight="semibold")
    axis.set_ylabel(
        "# of data points in training dataset", fontsize=18, weight="semibold"
    )
    return figure


def __get_time_stamp_for_file_name() -> str:
    """Get the time stamp for the file name"""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def __save_plot(
    figure: Figure,
    results_dataframe: pd.DataFrame,
    plot_type: str,
    chosen_metric: str,
    file_format="png",
) -> None:
    """Save a correlation plot."""
    dataset_name = __get_dataset_name(results_dataframe)
    time_stamp_string = __get_time_stamp_for_file_name()
    folder_location = f"plots/{dataset_name}/{chosen_metric}_correlation"
    os.makedirs(folder_location, exist_ok=True)

    filename = f"correlation_{plot_type}_{time_stamp_string}.{file_format}"
    counter = 1
    while os.path.exists(os.path.join(folder_location, filename)):
        counter += 1
        filename = (
            f"correlation_{plot_type}_{time_stamp_string}_{counter}.{file_format}"
        )

    filepath = os.path.join(folder_location, filename)
    logging.info(f"Saving correlation plot to {filepath}")
    figure.savefig(filepath, bbox_inches="tight", pad_inches=0.1)
    plt.close(figure.figure)


plot_correlation_Npoints_vs_MAPE = method_report(
    __plot_correlation_npoints_vs_metric, __save_plot
)
