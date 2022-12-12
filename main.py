#!/usr/bin/env python3

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Generator
from methods.predict import Predict
from data.dataset import Dataset, Result
from predictions.Prediction import PredictionData
from data.load import Load
from data.report import Report
from data.india_pollution import india_pollution, get_list_of_city_names
from data.stock_prices import stock_prices
from data.list_of_tuples import list_of_tuples
from data.airline_passengers import airline_passengers
from data.sun_spots import sun_spots
from data.data_from_csv import load_from_csv
from predictions.AR import ar
from predictions.MA import ma
from predictions.HoltWinters import holt_winters
from predictions.ARIMA import arima
from predictions.SARIMA import sarima
from predictions.linear_regression import linear_regression
from predictions.prophet import prophet
from predictions.FCNN import fcnn
from predictions.FCNN_embedding import fcnn_embedding
from predictions.SES import ses
from predictions.tsetlin_machine import tsetlin_machine
from measurements.get_metrics import get_metrics
from measurements.store_metrics import store_metrics
from plots.comparison_plot import comparison_plot
from plots.comparison_plot_multi import comparison_plot_multi
from plots.plot_results_in_heatmap import plot_results_in_heatmap

import time
import logging


__dataset_loaders: dict[str, Load[Dataset]] = {
    "india_pollution": india_pollution,
    "stock_prices": stock_prices,
    "list_of_tuples": list_of_tuples,
    "airline_passengers": airline_passengers,
    "sun_spots": sun_spots,
    "csv": load_from_csv,
}


__dataset_row_items: dict[str, list[str]] = {
    # from city Guhwati onwards
    "india_pollution": get_list_of_city_names()[:5],
    "stock_prices": ["JPM", "AAPL"],
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
    "MA": ma,
    "HoltWinters": holt_winters,
    "TsetlinMachine": tsetlin_machine,
}

__testset_size = 0.2


def load_dataset(dataset_name: str) -> list[Dataset]:
    """
    Load the given dataset.
    """
    if dataset_name not in __dataset_row_items:
        return [__dataset_loaders[dataset_name]()]
    else:
        return __dataset_loaders[dataset_name](__dataset_row_items[dataset_name])


def calculate_metrics(prediction: PredictionData):
    """
    Calculate the metrics for the given data and prediction.
    """
    return get_metrics(prediction)


def predict_measure_plot(data: Dataset, method_name: str) -> Report:
    """Generate a report for the given data and method."""

    start_time = time.time()
    print(f"Predicting {data.name}, specifically {data.subset_row_name} using {method_name}...")
    prediction = __predictors[method_name](data)
    metrics = calculate_metrics(prediction)

    training_index = data.values.index[
        : int(len(data.values.index) * (1 - __testset_size))
    ]
    comparison_plot(data.values.loc[training_index, :], prediction)

    return Report(start_time, method_name, data, prediction, metrics)


def __get_minimum_length_for_dataset(dataset: Dataset, method_name: str) -> int:
    """ Get the minimum length of the given dataset. """
    minimum_length = 0
    if dataset.time_unit == "days":
        minimum_length = 365*2.2
    elif dataset.time_unit == "weeks":
        minimum_length = 52*2.2
    elif dataset.time_unit == "months":
        minimum_length = 12*2.2
    elif dataset.time_unit == "years":
        minimum_length = 12*2.2
    if method_name == "SES":
        minimum_length = 20

    if method_name in ["MA", "AR", "ARIMA"]:
        minimum_length = int(minimum_length/2 *1.2)

    return int(minimum_length)




def generate_predictions(methods: list[str], datasets: list[str]) -> Generator[Report, None, None]:
    """
    Generate a report for each method and dataset combination.
    """
    
    for dataset_name in datasets:
        
        data_list = load_dataset(dataset_name)
        results_store = []
        for dataset in data_list:
            predictions_per_dataset = []
            reports_per_dataset = []
            training_index = dataset.values.index[: int(len(dataset.values.index) * (1 - __testset_size))]
            for method_name in methods:

                minimum_length = __get_minimum_length_for_dataset(dataset, method_name)
                if len(dataset.values) < int(minimum_length) and method_name in [ "SES", "HoltWinters", "SARIMA", "MA", "AR", "ARIMA"]:
                    print(f"Skipping {dataset.name} - {dataset.subset_row_name} for method {method_name} as at {len(dataset.values)} {dataset.time_unit} length it is too small.")
                    continue
                
                report = predict_measure_plot(dataset, method_name)
                store_metrics(report)

                predictions_per_dataset.append(report.prediction)
                reports_per_dataset.append(report)

            if len(predictions_per_dataset) > 0:
                comparison_plot_multi(dataset.values.loc[training_index, :], predictions_per_dataset)

            results_store.append(reports_per_dataset)
            yield reports_per_dataset
        
        logging.info(f"Plotting results for all datasets in {dataset_name}")
        plot_results_in_heatmap(results_store)




__datasets = [
     "india_pollution",
    # "stock_prices",
    # "airline_passengers",
    # "list_of_tuples",
    #"sun_spots",
    # "csv",
]


__methods = [
    # "AR",
    # "linear_regression",
    # "ARIMA",
    "HoltWinters",
    # "MA",
    # "Prophet",
    # "FCNN",
    # "FCNN_embedding",
    # "SARIMA",
    # "SES",
    # "TsetlinMachine",
]


for list_of_reports in generate_predictions(__methods, __datasets):
    print(list_of_reports)
