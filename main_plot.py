#!/usr/bin/env python3
if __name__ == "__main__":
    from typing import (
        Any,
        Callable,
        Dict,
        List,
        Optional,
        Tuple,
        Type,
        Union,
        Generator,
    )
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import TypeVar, List
from predictions.Prediction import PredictionData

from reports.report_loader import report_loader
from reports.report_loader import json_report_loader
from matplotlib import pyplot as plt
import seaborn as sns
import json
import gzip
from data.dataset import Dataset, Result
from data.report import Report
from methods.predict import Predict

from plots.plot_results_in_heatmap import plot_results_in_heatmap_from_csv
from plots.plot_results_in_box_plot import plot_results_in_boxplot_from_csv
from plots.comparison_plot_multi import comparison_plot_multi

from predictions.AR import ar
from predictions.MA import ma
from predictions.HoltWinters import holt_winters
from predictions.ARIMA import arima
from predictions.SARIMA import sarima
from predictions.auto_arima import auto_arima
from predictions.linear_regression import linear_regression
from predictions.prophet import prophet
from predictions.FCNN import fcnn
from predictions.FCNN_embedding import fcnn_embedding
from predictions.SES import ses
from predictions.tsetlin_machine import tsetlin_machine, tsetlin_machine_single

from data.list_of_tuples import list_of_tuples
from data.airline_passengers import airline_passengers
from data.sun_spots import sun_spots
from data.data_from_csv import load_from_csv
from data.load import Load

from data.india_pollution import (
    india_pollution,
    get_list_of_city_names,
    get_list_of_coastal_indian_cities,
)


from data.stock_prices import (
    stock_prices,
    get_a_list_of_value_stock_tickers,
    get_a_list_of_growth_stock_tickers,
)

__testset_size = 0.2

Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction", covariant=True)
ConfidenceInterval = TypeVar("ConfidenceInterval", covariant=True)
Title = TypeVar("Title", covariant=True)
Figure = TypeVar("Figure", covariant=True)


from methods.plot import Plot

__dataset_loaders: dict[str, Load[Dataset]] = {
    "India city pollution": india_pollution,
    "stock_prices": stock_prices,
    "list_of_tuples": list_of_tuples,
    "airline_passengers": airline_passengers,
    "sun_spots": sun_spots,
    "csv": load_from_csv,
}

# pick one dataset from the list only
__dataset = [
    # "India city pollution",
    "stock_prices",
    # "Airline passengers",
#    "airline_passengers",
    # "Straight line",
    # "Sunspots",
    # "CSV",
]


# choose the subset rows for the dataset to be plotted
__dataset_row_items: dict[str, list[str]] = {
    "India city pollution": get_list_of_city_names()[:7],  # get_list_of_city_names(),  # ["Ahmedabad", "Bengaluru", "Chennai"],
    "stock_prices": get_a_list_of_value_stock_tickers(),  # ["JPM", "AAPL", "MSFT"],# get_a_list_of_growth_stock_tickers()[:2],#get_a_list_of_value_stock_tickers(),
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

__plotters: dict[str, Plot[Data, Prediction, ConfidenceInterval, Title]] = {
    "heatmap": plot_results_in_heatmap_from_csv,
    "boxplot": plot_results_in_boxplot_from_csv,
    # "comparison": comparison_plot_multi,
}

__predictors: dict[str, Predict[Dataset, Result]] = {
    "linear_regression": linear_regression,
    "AR": ar,
    "ARIMA": arima,
    "Prophet": prophet,
    "FCNN": fcnn,
    "FCNN_embedding": fcnn_embedding,
    "SES": ses,
    "SARIMA": sarima,
    "auto_arima": auto_arima,
    "MA": ma,
    "HoltWinters": holt_winters,
    "TsetlinMachineMulti": tsetlin_machine,
    "TsetlinMachineSingle": tsetlin_machine_single,
}


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
    if dataset == "India city pollution":
        dataframe_of_results = dataframe_of_results.loc[
            dataframe_of_results["Dataset"] == "Indian city pollution"
        ]
    elif dataset == "stock_prices":
        dataframe_of_results = dataframe_of_results.loc[
            dataframe_of_results["Dataset"] == "Stock price"
        ]
    else:
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
        subset=["Dataset", "Topic", "Model"],
        keep="first",
        inplace=True,
        ignore_index=False,
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
    ["Dataset", "Topic", "Model", "MAE", "RMSE", "MAPE", "R Squared", "Filepath"]
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


# Creates a list of cities with R2<-10,if a city repeated twice (different methods),cancels the duplicate
mask_low_r_squared = filtered_dataframe.loc[
    filtered_dataframe["R2"] < -10
].drop_duplicates(subset=["subset_row"])["subset_row"]
# Filteres the above cities from the final list
filtered_dataframe_r_squared = filtered_dataframe[
    ~filtered_dataframe.subset_row.isin(mask_low_r_squared)
]


def load_dataset(dataset_name: str) -> list[Dataset]:
    """
    Load the given dataset.
    """
    if dataset_name not in __dataset_row_items:
        return [__dataset_loaders[dataset_name]()]
    else:
        return __dataset_loaders[dataset_name](__dataset_row_items[dataset_name])


def generate_predictions_from_zip_json(data: pd.DataFrame) -> PredictionData:
    """
    Generates PredictionData from zip-json
    """
    json_data_store = []
    for i in range(len(data["Filepath"].index)):
        if not np.isnan(data["MAPE"][i]):
            json_data = json_report_loader(data["Filepath"][i])
            json_data_store.append(json_data)
    return json_data_store


for dataset_name in __dataset:
    # TODO: decide if need to run on lots of datasets or just one

    data_list = load_dataset(dataset_name)
    for dataset in data_list:
        training_index = dataset.values.index[
            : int(len(dataset.values.index) * (1 - __testset_size))
        ]
        id_count = np.where(filtered_dataframe['subset_row']==dataset.subset_row_name)
        id_count = np.array(id_count).ravel()

        prediction_per_city = []
        for i in range (len(id_count)):
            prediction = generate_predictions_from_zip_json(filtered_dataframe)[id_count[i]]
            prediction_per_city.append(prediction)

        comparison_plot_multi(dataset.values.loc[training_index, :], prediction_per_city)



for plotter_name, plotter in __plotters.items():
    print(f"Plotting {plotter_name} for dataset: {dataset_name}")
    if plotter_name == "boxplot":
        plotter(filtered_dataframe, dataset_name)
    # elif plotter_name == "comparison":
    #     plotter(filtered_dataframe, dataset_name)
    # else:
    #     plotter(filtered_dataframe_r_squared, dataset_name)
