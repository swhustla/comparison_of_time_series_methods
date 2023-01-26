#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
from typing import TypeVar, List

from reports.report_loader import report_loader

from plots.plot_results_in_heatmap import plot_results_in_heatmap_from_csv

Data = TypeVar("Data", contravariant=True)

from methods.plot import Figure, Plot

# pick one dataset from the list only
__dataset = [
    "Indian city pollution",
    #  "Stock prices",
    # "Airline passengers",
    # "Straight line",
    # "Sunspots",
    # "CSV",
]


__dataset_row_items: dict[str, list[str]] = {
    # take first 3 from list of cities
    "Indian city pollution": ["Ahmedabad", "Bengaluru", "Chennai"],
    "Stock prices": ["JPM", "AAPL", "MSFT"],
}


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

__plotters: dict[str, Plot[Data]] = {"heatmap": plot_results_in_heatmap_from_csv}


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
    dataframe_of_results = dataframe_of_results[
        dataframe_of_results["Dataset"]
        == dataset
        & dataframe_of_results["Topic"].isin(topics)
        & dataframe_of_results["Method"].isin(methods)
    ]

    dataframe_of_results = dataframe_of_results.sort_values(
        by=["End Time"], ascending=False
    )
    dataframe_of_results = dataframe_of_results.drop_duplicates(
        subset=["Dataset", "Topic", "Method"], keep="first"
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
    ["Dataset", "Topic", "Method", "MAE", "RMSE", "MAPE", "R2"]
]

# rename the columns to be used in the heatmap
filtered_dataframe = filtered_dataframe.rename(
    columns={
        "Method": "method",
        "Dataset": "dataset",
        "Topic": "subset_row",
    }
)
dataset_name = __dataset[0]
# plot the heatmap using the csv file

for plotter_name, plotter in __plotters.items():
    print(f"Plotting {plotter_name} for dataset: {dataset_name}")

    plotter(filtered_dataframe)
