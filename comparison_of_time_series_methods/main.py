#!/usr/bin/env python3
if __name__ == "__main__":
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)

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
    from methods.predict import Predict
    from data.dataset import Dataset, Result
    from predictions.Prediction import PredictionData
    from data.load import Load
    from data.report import Report
    from data.india_pollution import (
        india_pollution,
        get_list_of_city_names,
        get_list_of_coastal_indian_cities,
        get_cities_from_geographical_region,
        get_city_list_by_tier,
    )
    from data.stock_prices import (
        stock_prices,
        get_a_list_of_value_stock_tickers,
        get_a_list_of_growth_stock_tickers,
    )
    from data.list_of_tuples import list_of_tuples
    from data.airline_passengers import airline_passengers
    from data.sun_spots import sun_spots
    from data.data_from_csv import load_from_csv
    from data.electricity_consumption import energy_demand
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
    from measurements.get_metrics import get_metrics
    from measurements.store_metrics import store_metrics
    from plots.comparison_plot import comparison_plot
    from plots.comparison_plot_multi import comparison_plot_multi
    from plots.plot_results_in_heatmap import plot_results_in_heatmap
    from plots.plot_results_in_scatter_plot import plot_results_in_scatter_plot

    import time
    import logging
    import numpy as np
    import pandas as pd

    __dataset_loaders: dict[str, Load[Dataset]] = {
        "india_pollution": india_pollution,
        "stock_prices": stock_prices,
        "list_of_tuples": list_of_tuples,
        "airline_passengers": airline_passengers,
        "sun_spots": sun_spots,
        "csv": load_from_csv,
        "electricity_consumption": energy_demand,
    }

    __dataset_row_items: dict[str, list[str]] = {
        # take first 3 from list of cities
        "stock_prices":   get_a_list_of_growth_stock_tickers()[:2],#get_a_list_of_value_stock_tickers(),
        "india_pollution": get_cities_from_geographical_region("Indo-Gangetic Plain"),
    }

    __dataset_group_titles: dict[str, str] = {
        "india_pollution": "Cities on the Indo-Gangetic Plain in India",
        "stock_prices": "Growth stocks",
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
        logging.info(
            f"Predicting {data.name}, specifically {data.subset_row_name} using {method_name}..."
        )
        prediction = __predictors[method_name](data)
        metrics = calculate_metrics(prediction)

        # if data is a series, then we need to convert it to a dataframe
        if isinstance(data.values, pd.Series):
            data.values = data.values.to_frame()

        training_data = data.values.iloc[
            : int(len(data.values.index) * (1 - __testset_size)), :
        ]

        comparison_plot(
            training_data,
            prediction,
        )
        datestring_today = time.strftime("%Y-%m-%d")
        filepath = f"reports/full_data/{data.name}_{data.subset_row_name}_{method_name}_{datestring_today}.json.gz"
        end_time = time.time()
        logging.info(f"Saving report to {filepath}...")
        return Report(
            start_time,
            method_name,
            data,
            prediction,
            metrics,
            filepath=filepath,
            end_time=end_time,
        )

    def __calculate_minimum_length_given_periodicity(periodicity: int) -> int:
        """Calculate the minimum length of the dataset given the periodicity."""
        return int(periodicity * 2 * 1.25)

    def __get_minimum_length_for_dataset(dataset: Dataset, method_name: str) -> int:
        """Get the minimum length of the given dataset."""
        minimum_length = 0
        if dataset.time_unit == "days":
            minimum_length = __calculate_minimum_length_given_periodicity(365)
        elif dataset.time_unit == "weeks":
            minimum_length = __calculate_minimum_length_given_periodicity(52)
        elif dataset.time_unit == "months":
            minimum_length = __calculate_minimum_length_given_periodicity(12)
        elif dataset.time_unit == "years":
            minimum_length = __calculate_minimum_length_given_periodicity(11)
        if method_name == "SES":
            minimum_length = 20

        if method_name in ["MA", "AR", "ARIMA"]:
            minimum_length = int(minimum_length / 2 * 1.2)

        return int(minimum_length)

    def __check_to_convert_to_weekly_data(data: Dataset) -> Dataset:
        """Convert the data to weekly data if it is in daily data, and if there are enough data points.
        Ideally we need at least 2 years of data to make a weekly prediction."""
        if data.time_unit == "days" and len(data.values.index) > 365 * 2:
            data.values = data.values.resample("W").mean()
            data.time_unit = "weeks"

        return data

    def generate_predictions(
        methods: list[str], datasets: list[str]
    ) -> Generator[Report, None, None]:
        """
        Generate a report for each method and dataset combination.
        """
        number_of_methods = len(list(methods))

        for dataset_name in datasets:

            data_list = load_dataset(dataset_name)
            results_store = []

            print(f"types: {type(results_store)} {type(data_list)}")
            for dataset in data_list:
                predictions_per_dataset = []
                reports_per_dataset = []
                # to store R2
                metrics_stat = []

                training_index = dataset.values.index[
                    : int(len(dataset.values.index) * (1 - __testset_size))
                ]
                for method_name in methods:

                    if method_name in ["SARIMA"] and dataset.time_unit == "days":
                        dataset = __check_to_convert_to_weekly_data(
                            dataset
                        )  # convert to weekly data if SARIMA is used
                        training_index = dataset.values.index[
                            : int(len(dataset.values.index) * (1 - __testset_size))
                        ]  # update training index accordingly

                    minimum_length = __get_minimum_length_for_dataset(
                        dataset, method_name
                    )
                    if len(dataset.values) < int(minimum_length) and method_name in [
                        "SES",
                        "HoltWinters",
                        "SARIMA",
                        "MA",
                        "AR",
                        "ARIMA",
                    ]:
                        print(
                            f"Skipping {dataset.name} - {dataset.subset_row_name} for method {method_name} as at {len(dataset.values)} {dataset.time_unit} length it is too small. \n Minimum length is {minimum_length} {dataset.time_unit}"
                        )
                        continue

                    report = predict_measure_plot(dataset, method_name)
                    store_metrics(report)

                    predictions_per_dataset.append(report.prediction)
                    reports_per_dataset.append(report)

                    # to store R2
                    metrics = calculate_metrics(report.prediction)["r_squared"]
                    metrics_stat.append(metrics)

                if len(predictions_per_dataset) > 0:
                    comparison_plot_multi(
                        dataset.values.loc[training_index, :],
                        predictions_per_dataset,
                    )

                # appends only the cities/stocks where R2 > 10
                if (np.array(metrics_stat) > -10).all() or (
                    dataset_name == "stock_prices"
                ):
                    results_store.append(reports_per_dataset)

                # plot into a scatter plot if at least 3 methods are used
                logging.info(f"Number of methods: {number_of_methods}")
                if number_of_methods > 2:
                    logging.info(f"Plotting results into scatter plot for {dataset.name} - {dataset.subset_row_name}...")
                    plot_results_in_scatter_plot(reports_per_dataset)

                yield reports_per_dataset

            if len(results_store) > 1 and number_of_methods > 1:
                logging.info(
                    f"Plotting results for all {len(results_store)} datasets in {dataset_name} for {number_of_methods} methods..."
                )
                plot_results_in_heatmap(
                    results_store, __dataset_group_titles[dataset_name]
                )
                logging.info(
                    f"Plotting results for all {len(results_store)} datasets in {dataset_name} - done"
                )

    __datasets = [
        # "india_pollution",
        #  "stock_prices",
        "airline_passengers",
        # "list_of_tuples",
        #  "sun_spots",
        # "csv",
        # "electricity_consumption",
    ]

    __methods = [
        # "AR",
        "linear_regression",
        # "ARIMA",
        # "HoltWinters",
        # "MA",
        # "Prophet",
        # "FCNN",
        # "FCNN_embedding",
        # "SARIMA",
        # "auto_arima",
        # "SES",
        # "TsetlinMachineSingle",
        # "TsetlinMachineMulti",
    ]


    for list_of_reports in generate_predictions(__methods, __datasets):
        print(list_of_reports)

