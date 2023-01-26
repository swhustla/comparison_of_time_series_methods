"""Apply STL decomposition to the data.

Seasonal decomposition is a time series analysis technique that
decomposes a time series into three components: trend, seasonality,
and noise. This is useful for identifying trends and seasonality in
the data, and for removing them to get a better understanding of the
noise in the data.

This function uses the statsmodels STL decomposition function to
perform the decomposition. The function is applied to each column of
the data, and the results are stored in a new dataset.

"""

import os
import logging
import pandas as pd
from statsmodels.tsa.seasonal import STL, DecomposeResult
from data.dataset import Dataset

import matplotlib.pyplot as plt

from methods.seasonal_decompose import seasonal_decompose as method

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def __get_seasonal_period(data: Dataset) -> int:
    """Returns the seasonal period"""
    if data.time_unit == "months":
        return 12
    elif data.time_unit == "weeks":
        return 52
    elif data.time_unit == "days":
        return 365
    elif data.time_unit == "years":
        return 11
    else:
        return 1


def __determine_if_seasonal_with_acf(dataset: Dataset) -> bool:
    """Determines if the data has a seasonal component
    overriden by the Dataset metadata"""
    if dataset.seasonality is None:

        if type(dataset.values) is pd.DataFrame:
            series = pd.Series(
                dataset.values[dataset.subset_column_name],
                index=dataset.values.index,
            )
        else:
            series = dataset.values.observed
        series.dropna(inplace=True)

        return (
            series.autocorr(
                lag=__get_seasonal_period(dataset),
            )
            > 0.5
        )
    else:
        return dataset.seasonality


def __store_plot_of_decomposition_results(
    data: Dataset, decompose_result: DecomposeResult, type: str = "old"
) -> None:
    """Stores a plot of the decomposition results"""
    logging.info("Storing plot of decomposition results")
    plt.rcParams.update({"figure.figsize": (10, 9)})
    fig = decompose_result.plot()

    file_path = f"{PROJECT_FOLDER}/plots/{data.name}/{data.subset_row_name}/STL/{type}_seasonal_decomposition.png"

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

    fig.savefig(file_path)


def __seasonal_decompose_data(data: Dataset) -> Dataset:
    """Perform STL decomposition on the data."""
    logging.info("Performing STL decomposition")
    seasonal_component_present = __determine_if_seasonal_with_acf(data)

    logging.info(f"Seasonal component present: {seasonal_component_present} for {data.name}")

    if seasonal_component_present:
        seasonal_period = __get_seasonal_period(data)
    else:
        seasonal_period = 1

    logging.info(f"Seasonal period: {seasonal_period}")

    decomposition_stl = STL(
        data.values[data.subset_column_name],
        period=seasonal_period,
        robust=True,
    ).fit()

    __store_plot_of_decomposition_results(data, decomposition_stl, "STL")

    return Dataset(
        name=data.name,
        time_unit=data.time_unit,
        values=decomposition_stl,
        number_columns=data.number_columns,
        subset_row_name=data.subset_row_name,
        subset_column_name=data.subset_column_name,
        seasonality=seasonal_component_present,
    )



seasonal_decompose = method(__seasonal_decompose_data)