"""Plot a correlation plot showing the correlation between the number of data points in the training set and the metric chosen. The metric chosen can be MAE, RMSE, R2 or MAPE. The plot is saved in the folder plots/correlation_Npoints_vs_metric."""
import os
import logging
import datetime

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from typing import List, Tuple, Optional, Callable, TypeVar, Generator, Union
import pandas as pd

Figure = TypeVar("Figure")
from predictions.Prediction import PredictionData
from plots.color_map_by_method import __color_map_by_method_dict
from methods.plot_correlation_Npoints import (
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


def __plot_correlation_Npoints_vs_metric(
    training_data: List[pd.DataFrame],
    results_dataframe: pd.DataFrame,
    chosen_metric: str = "MAE",
    group_name: Optional[str] = None,
) -> Figure:
    """Plot correlation N data points in training set vs metric."""
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.set_title(__get_title(results_dataframe, chosen_metric, group_name))
    # condensed distance matrix must contain only finite values
    results_dataframe[chosen_metric] = results_dataframe[chosen_metric].replace(
        [np.inf, -np.inf], np.nan
    )

    pivoted_dataframe = results_dataframe.pivot(
        columns="method", index="subset_row", values=chosen_metric
    ).reset_index()
    # condensed distance matrix must contain only finite values
    pivoted_dataframe = pivoted_dataframe.replace([np.inf, -np.inf], np.nan)

    # deal with the case where there are no results for a method
    pivoted_dataframe = pivoted_dataframe.fillna(0)
    data_lenght = []
    for data in training_data:
        data_lenght.append(len(data))
    pivoted_dataframe_Ndata = pivoted_dataframe.copy()
    pivoted_dataframe_Ndata["Ndata"] = data_lenght

    if chosen_metric == "R2":
        # create more ticks on the x axis
        sns.scatterplot(
            data=pivoted_dataframe,
            x=pivoted_dataframe.min(axis=1, numeric_only=True),
            y=data_lenght,
            ax=axis,
            hue=pivoted_dataframe["subset_row"],
            palette=sns.color_palette("hls", len(data_lenght)),
            marker="X",
            s=200,
        )
        # add labels and legend to plot
        for index, row in pivoted_dataframe.iterrows():
            x = pd.to_numeric(row, errors="coerce").min()
            y = data_lenght[index]
            name = row["subset_row"]
            bbox_props = dict(
                boxstyle="round,pad=0.5", fc="white", ec="gray", lw=1, alpha=0.5
            )
            axis.text(x, y, name, ha="center", va="center", bbox=bbox_props)
            axis.legend(fontsize="small", title_fontsize="40", loc="upper right")
        # Add horizontal and vertical lines
        axis.axhline(y=100, color="gray", linestyle="--")
        axis.axvline(x=-10, color="gray", linestyle="--")
        # set x-axis range
        xmin = pivoted_dataframe.min(axis=1, numeric_only=True).min()
        xmax = pivoted_dataframe.min(axis=1, numeric_only=True).max()

        # set x-axis tick increment
        xtick_increment = 0.1

        # create x-axis ticks
        xticks = [
            xmin + i * xtick_increment
            for i in range(int((xmax - xmin) / xtick_increment) + 1)
        ]
        axis.tick_params(
            axis="both", which="major", labelsize=17, width=2, labelrotation=50
        )
        # set x-axis ticks
        axis.set_xticks(xticks)
        # set x-axis scale
        axis.set_xscale("symlog")
        axis.set_xlim(1.1 * xmin, 1.0)
    else:
        sns.scatterplot(
            data=pivoted_dataframe,
            x=pivoted_dataframe.max(axis=1, numeric_only=True),
            y=data_lenght,
            ax=axis,
            hue=pivoted_dataframe["subset_row"],
            palette=sns.color_palette("hls", len(data_lenght)),
            marker="X",
            s=200,
        )
        # add labels and legend to plot
        for index, row in pivoted_dataframe.iterrows():
            x = pd.to_numeric(row, errors="coerce").max()
            y = data_lenght[index]
            name = row["subset_row"]
            bbox_props = dict(
                boxstyle="round,pad=0.5", fc="white", ec="gray", lw=1, alpha=0.5
            )
            axis.text(x, y, name, ha="center", va="center", bbox=bbox_props)
            axis.legend(fontsize="small", title_fontsize="40", loc="upper right")
        # add horizontal line
        axis.axhline(y=100, color="gray", linestyle="--")

    # plot all data points of each method
    # axis = sns.scatterplot(
    #     x="value",
    #     hue="series",
    #     y="Ndata",
    #     data=pivoted_dataframe_Ndata.melt(
    #         value_vars=[
    #             "AR",
    #             "ARIMA",
    #             "HoltWinters",
    #             "MA",
    #             "Prophet",
    #             "SARIMA",
    #             "SES",
    #             "linear_regression",
    #         ],
    #         id_vars="Ndata",
    #         var_name="series",
    #     ),
    # )

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
    __plot_correlation_Npoints_vs_metric, __save_plot
)
