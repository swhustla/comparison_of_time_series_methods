#!/usr/bin/env python3

from methods.predict import Predict
from data.Data import Dataset, Result
from predictions.Prediction import PredictionData
from data.load import Load
from data.report import Report
from data.india_pollution import india_pollution
from data.stock_prices import stock_prices
from data.list_of_tuples import list_of_tuples
from data.airline_passengers import airline_passengers
from predictions.ARIMA import arima
from predictions.SARIMA import sarima
from predictions.linear_regression import linear_regression
from predictions.prophet import prophet
from predictions.FCNN import fcnn
from predictions.FCNN_embedding import fcnn_embedding
from predictions.SES import ses
from measurements.get_metrics import get_metrics
from measurements.store_metrics import store_metrics
from plots.comparison_plot import comparison_plot
import time


__dataset_loaders: dict[str, Load[Dataset]] = {
    "india_pollution": india_pollution,
    "stock_prices": stock_prices,
    "list_of_tuples": list_of_tuples, 
    "airline_passengers": airline_passengers,
}



__predictors: dict[str, Predict[Dataset, Result]] = {
    "linear_regression": linear_regression,
    "ARIMA": arima,
    "Prophet": prophet,
    "FCNN": fcnn,
    "FCNN_embedding": fcnn_embedding,
    "SES": ses,
    "SARIMA": sarima,
}

__testset_size = 0.2


def load_dataset(dataset_name: str):
    """
    Load the given dataset.
    """
    return __dataset_loaders[dataset_name]()


def calculate_metrics(prediction: PredictionData):
    """
    Calculate the metrics for the given data and prediction.
    """
    return get_metrics(prediction)

def predict_measure_plot(data: Dataset, method_name: str) -> Report:
    """Generate a report for the given data and method."""

    start_time = time.time()
    print(f"Predicting {data.name} using {method_name}...")
    prediction  = __predictors[method_name](data)
    metrics = calculate_metrics(prediction)

    training_index = data.values.index[:int(len(data.values.index) * (1 - __testset_size))]
    comparison_plot(data.values.loc[training_index, :], prediction)
 
    return Report(start_time, method_name, data, prediction, metrics)


def generate_predictions(methods: list[str], datasets: list[str]):
    """
    Generate predictions for the given dataset using the given methods.
    """
    for method_name in methods:
        for dataset_name in datasets:
            data = load_dataset(dataset_name)
            if method_name in ["SARIMA"]:
                if data.time_unit == "days":
                    print(f"Skipping {method_name} on {data.name} because it does not support daily data.")
                    continue
            report = predict_measure_plot(data, method_name)
            store_metrics(report)
            yield report



__datasets = [
    "india_pollution", 
    "stock_prices", 
    "airline_passengers",
    "list_of_tuples"
    ]

__methods = [
    "linear_regression", 
    "ARIMA",
    "Prophet",
    "FCNN",
    "FCNN_embedding",
    "SES",
    "SARIMA",
    # "LSTM"
    # "MSTSD"
    ]


for report in generate_predictions(__methods, __datasets):
    print(report)
