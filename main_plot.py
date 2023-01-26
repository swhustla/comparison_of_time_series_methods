#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
from typing import TypeVar, List

from reports.report_loader import report_loader

from plots.plot_results_in_heatmap import plot_results_in_heatmap_from_csv

from data.india_pollution import (
        india_pollution,
        get_list_of_city_names,
        get_list_of_coastal_indian_cities,
    )

Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction", covariant=True)
ConfidenceInterval = TypeVar("ConfidenceInterval", covariant=True)
Title = TypeVar("Title", covariant=True)
Figure = TypeVar("Figure", covariant=True)

from methods.plot import Plot

# pick one dataset from the list only
__dataset = [
    "Indian city pollution",
    #  "Stock prices",
    # "Airline passengers",
    # "Straight line",
    # "Sunspots",
    # "CSV",
]


# choose the subset rows for the dataset to be plotted
__dataset_row_items: dict[str, list[str]] = {
    "Indian city pollution": get_list_of_city_names(),#["Ahmedabad", "Bengaluru", "Chennai"],
    "Stock prices": ["JPM", "AAPL", "MSFT"],
}

# pick at least 2 methods from the list
__methods = [
    "AR",
    "linear_regression",
    "ARIMA",
    "HoltWinters",
    "MA",
    "Prophet",
    # "FCNN",
    # "FCNN_embedding",
    "SARIMA",
    # "auto_arima"
    "SES",
    # "TsetlinMachine",
]

__plotters: dict[str, Plot[Data, Prediction, ConfidenceInterval, Title]] = {"heatmap": plot_results_in_heatmap_from_csv}


def filter_dataframe_by_dataset_method_and_subset(
    dataset: str, topics: List[str], methods: List[str]
) -> pd.DataFrame:
    """Method to filter the report by dataset, method and subset row.
    Columns of reports/summary_report.csv:
    Dataset, Topic, Method, Start Time, End Time, Training Time, Prediction Time, MAE, RMSE, MAPE, R2, Prediction
    Ensure the most recent run for a given dataset, method and subset row is used only.
    """
    dataframe_of_results = report_loader()
    if dataframe_of_results is None:
        return pd.DataFrame()
    dataframe_of_results = dataframe_of_results.loc[
        dataframe_of_results["Dataset"] == dataset
    ]
    dataframe_of_results = dataframe_of_results.loc[
        dataframe_of_results["Topic"].isin(topics)
    ]
    dataframe_of_results = dataframe_of_results.loc[
        dataframe_of_results["Model"].isin(methods)
    ]

    # sort the dataframe by the index, most recent first

    dataframe_of_results = dataframe_of_results.sort_index(ascending=False)

    # drop the duplicates, check the columns to be used in duplicate removal
    dataframe_of_results.drop_duplicates(
        subset=["Dataset", "Topic", "Model"], keep="first", inplace=True, ignore_index=False
    )
    
    return dataframe_of_results


print(f"Plotting results for dataset: {__dataset[0]}")

print(f"...and methods: {__methods}")

print(f"......and subset rows: { __dataset_row_items[__dataset[0]]}")
filtered_dataframe = filter_dataframe_by_dataset_method_and_subset(
    __dataset[0], __dataset_row_items[__dataset[0]], __methods
)
# re-format the report to be used in the heatmap
filtered_dataframe = filtered_dataframe[
    ["Dataset", "Topic", "Model", "MAE", "RMSE", "MAPE", "R Squared"]
]

# rename the columns to be used in the heatmap
filtered_dataframe = filtered_dataframe.rename(
    columns={
        "Model": "method",
        "Dataset": "dataset",
        "Topic": "subset_row",
        "R Squared": "R2",
    }
)
dataset_name = __dataset[0]


#filter for R2 > -10
name_cities =  filtered_dataframe.loc[filtered_dataframe['R2'] < -10].drop_duplicates(
        subset=["subset_row"]
    )["subset_row"]
filtered_dataframe_R2 = filtered_dataframe[~filtered_dataframe.subset_row.isin(name_cities)]

# plot the heatmap using the csv file
for plotter_name, plotter in __plotters.items():
    print(f"Plotting {plotter_name} for dataset: {dataset_name}")

    plotter(filtered_dataframe_R2, dataset_name)
