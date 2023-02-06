"""Plot a heatmap and box plot showing the accuracy of lots of time series prediction methods 
on lots of different time series"""
import os
import logging
import datetime

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from typing import List, Tuple, Optional, Callable, TypeVar, Generator
import pandas as pd

Figure = TypeVar("Figure")
from predictions.Prediction import PredictionData
from plots.color_map_by_method import __color_map_by_method_dict

from data.report import Report


from methods.plot_results_in_box_plot import plot_results_in_boxplot_from_csv as method_report_from_csv



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


def __get_title(results_dataframe: pd.DataFrame, chosen_metric: str, group_name: str) -> str:
    """Get the title of the plot"""
    if group_name is None:
        return f"Box plot {chosen_metric} for {__get_dataset_name(results_dataframe)} for {results_dataframe['method'].unique().size} predictive methods on {results_dataframe['subset_row'].unique().size} datasets"
    else:
        return f"Box plot {chosen_metric} for {group_name} with {results_dataframe['method'].unique().size} predictive methods"



def add_median_labels(ax, fmt='.1f'):
    """Credits: https://stackoverflow.com/a/63295846/4865723
    """
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center',
                       fontweight='bold', color='white',fontsize=12)
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])


def __plot_boxplot_by_method(
    results_dataframe: pd.DataFrame, chosen_metric: str = "MAE", group_name: Optional[str] = None
) -> Figure:
    """Plot the results in a boxplot"""
    logging.info(f"Plotting {chosen_metric} boxplot")

    title = __get_title(results_dataframe, chosen_metric, group_name)
    color_map = __color_map_by_method_dict
    figure, axis = plt.subplots(figsize=(15, 9))

    results_dataframe[chosen_metric] = results_dataframe[chosen_metric].replace([np.inf, -np.inf], np.nan)

    pivoted_dataframe = results_dataframe.pivot(columns="method", index="subset_row", values=chosen_metric)
    # condensed distance matrix must contain only finite values
    pivoted_dataframe = pivoted_dataframe.replace([np.inf, -np.inf], np.nan)

    #deal with the case where there are no results for a method
    pivoted_dataframe = pivoted_dataframe.fillna(0)
    mask_for_missing_values = pivoted_dataframe == 0

    box_plot = sns.boxplot(
        x="method",
        y="MAPE", 
        data=results_dataframe,
        palette=color_map,
        linewidth=3,)

    sns.scatterplot(
        data=results_dataframe,
        x=results_dataframe["method"],
        y=results_dataframe["MAPE"],
        hue="method",
        ax=axis,
        palette=color_map,
        s=100,
    )
    add_median_labels(box_plot)

    plt.tick_params(axis='both', which='major', labelsize=17, width=2)
    axis.set_title(title,fontsize=18)
    plt.xticks(rotation=50, weight = 'normal')
    plt.yticks(weight = 'normal')
    plt.legend(fontsize='x-large', title_fontsize='40',loc='upper right')
    if __get_dataset_name(results_dataframe) == 'Stock price':
        axis.set_ylim(0, 70)
    else:
        axis.set_ylim(0, 150)
    axis.set_xlabel("Method",fontsize=18,weight = 'semibold')
    axis.set_ylabel(chosen_metric,fontsize=18,weight = 'semibold')
    return figure

def __plot_boxplot_by_city(
    results_dataframe: pd.DataFrame, chosen_metric: str = "MAE", group_name: Optional[str] = None
) -> Figure:
    """Plot the results in a boxplot"""
    logging.info(f"Plotting {chosen_metric} boxplot")

    title = __get_title(results_dataframe, chosen_metric, group_name)
    __get_dataset_name(results_dataframe)
    color_map = __color_map_by_method_dict
    figure, axis = plt.subplots(figsize=(15, 9))

    results_dataframe[chosen_metric] = results_dataframe[chosen_metric].replace([np.inf, -np.inf], np.nan)

    pivoted_dataframe = results_dataframe.pivot(columns="method", index="subset_row", values=chosen_metric)
    # condensed distance matrix must contain only finite values
    pivoted_dataframe = pivoted_dataframe.replace([np.inf, -np.inf], np.nan)

    #deal with the case where there are no results for a method
    pivoted_dataframe = pivoted_dataframe.fillna(0)
    mask_for_missing_values = pivoted_dataframe == 0

    box_plot = sns.boxplot(
        x="subset_row", 
        y="MAPE", 
        data=results_dataframe,
        linewidth=3,)
    add_median_labels(box_plot)

    sns.scatterplot(
        data=results_dataframe,
        x=results_dataframe["subset_row"],
        y=results_dataframe["MAPE"],
        hue="method",
        ax=axis,
        palette=color_map,
        s=100,
    )
    
    plt.tick_params(axis='both', which='major', labelsize=17, width=2)
    plt.xticks(rotation=50,weight ='normal')
    plt.yticks(weight ='normal')
    plt.legend(fontsize='x-large', title_fontsize='40',loc='upper right')
    if __get_dataset_name(results_dataframe) == 'Stock price':
        axis.set_xlabel("Stock price",fontsize=18,weight = 'semibold')
        axis.set_ylim(0, 70)
    else:
        axis.set_xlabel("City",fontsize=18,weight = 'semibold')
        axis.set_ylim(0, 150)
    axis.set_title(title,fontsize=18)
    axis.set_ylabel(chosen_metric,fontsize=18,weight = 'semibold')
    return figure

def __get_time_stamp_for_file_name() -> str:
    """Get the time stamp for the file name"""
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def __save_plot_boxplot(figure: Figure, dataset_name: str, plot_type: str ,chosen_metric: str) -> None:
    """Save the plot"""
    time_stamp_string = __get_time_stamp_for_file_name()
    folder_location = f"plots/{dataset_name}/{chosen_metric}_boxplot"
    logging.info(f"Saving box_plot to {folder_location}/boxplot_{plot_type}_{time_stamp_string}.png")
    if not os.path.exists(f"{folder_location}"):
        os.makedirs(f"{folder_location}")
    
    figure.savefig(
        f"{folder_location}/box_plot_{plot_type}.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )


plot_results_in_boxplot_from_csv = method_report_from_csv(
    __plot_boxplot_by_method, __plot_boxplot_by_city, __save_plot_boxplot
    )