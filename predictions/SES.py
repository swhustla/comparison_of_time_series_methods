""" 

Simple Exponential Smoothing (SES)

It was first introduced by Holt in 1957. It is a naive approach where the observations at
preceding time periods are weighted with a single smoothing parameter to forecast the
observations at the next time period.

The weights decrease exponentially as observations come from further in the past,
the smallest weights are associated with the oldest observations.

The weights are calculated using an exponential function. The exponential function is defined by
the smoothing parameter, alpha. The smoothing parameter determines the amount of exponential
smoothing. A larger value of alpha results in more exponential smoothing.

The exponential smoothing technique can be applied to both univariate and multivariate time series.

SES assumes that the data has be detrended and seasonaility has been removed, so we will use the 
seasonal_decompose to remove these from our data.

The process of seasonal decomposition involves thinking of a series as a combination of level, trend,
seasonality, and noise components. It is a statistical procedure that explicitly models the
decomposition of a time series into these components.

STL (Seasonal and Trend decomposition using Loess) is a generalization of the classical method of
seasonal decomposition. It is a robust method that works well with noisy data and is able to
handle missing values.

The benefit of using the STL method for prediction is that it is able to handle missing values
and noisy data.

Additive or multiplicative?
===========================
- Additive: If the seasonal variations are roughly constant through the 
series.
- Multiplicative: If the seasonal variations are changing proportional 
to the level of the series.

Multiplicative is more common with economic time series, because the 
seasonal variations are usually proportional to the level of the series.



How to determine if the series has a trend component?
===================================================
- If the series has a trend component, then the trend component should 
be removed before applying SES.
With ACF (Autocorrelation Function) we can determine if the series has
a trend component. If the ACF has a significant lag, then the series
has a trend component.
We can use the seasonal_decompose function to remove the trend component.


How to determine if the series has a seasonal component?
=======================================================
- If the series has a seasonal component, then the seasonal component
should be removed before applying SES.
Again using ACF, with a significant lag of 12 (for the case of months), we 
can determine if the series has a seasonal component.
If the ACF has a significant lag, then the series has a seasonal component.
We can use the seasonal_decompose function to remove the seasonal component.

How to detect if the seasonal component is additive or multiplicative?
=====================================================================
- If the seasonal component is multiplicative, the seasonal index will
increase or decrease at a constant rate over time. By dividing the raw
data by the moving average, and then subtracting it also we can visually determine
if the seasonal component is additive or multiplicative.
To determine this algorithmically, in Python, without visualizing the data, or using the
seasonal_decompose function, we can use the ACF to determine if the seasonal component
is additive or multiplicative, in the following way:
- If the ACF has a significant lag of 12, then the seasonal component is multiplicative.
- If the ACF has a significant lag of 1, then the seasonal component is additive.


Adding the trend and seasonal components back to the forecast
=============================================================
- If the series has a trend component, then the trend component should
be added back to the forecast. Likewise, if the series has a seasonal
component, then the seasonal component should be added back also.



"""
import os
from typing import TypeVar, Generic, List, Tuple, Dict
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from plots.color_map_by_method import get_color_map_by_method
from data.seasonal_decompose import seasonal_decompose as seasonal_decompose_data


import logging

# turn on logging

logging.basicConfig(level=logging.INFO)

from data.dataset import Dataset, Result
from predictions.Prediction import PredictionData

from methods.SES import ses as method


Model = TypeVar("Model")

__alpha = 0.2

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def __number_of_steps(data: Dataset) -> int:
    return int(len(data.values) * 0.2)


def __get_training_set(data: Dataset) -> Dataset:
    return Dataset(
        name=data.name,
        values=data.values[: -__number_of_steps(data)],
        number_columns=data.number_columns,
        subset_column_name=data.subset_column_name,
        time_unit=data.time_unit,
        subset_row_name=data.subset_row_name,
        seasonality=data.seasonality,
    )


def __get_test_set(data: Dataset) -> Dataset:
    return Dataset(
        name=data.name,
        values=data.values[-__number_of_steps(data) :],
        number_columns=data.number_columns,
        subset_column_name=data.subset_column_name,
        time_unit=data.time_unit,
        subset_row_name=data.subset_row_name,
        seasonality=data.seasonality,
    )


def __determine_if_trend_with_acf(decomposed_dataset: Dataset) -> bool:
    """Determines if the data has a trend component"""
    logging.info(f"Determining if {decomposed_dataset.name} has a trend component")
    if type(decomposed_dataset.values) is pd.DataFrame:
        series = pd.Series(
            decomposed_dataset.values[decomposed_dataset.subset_column_name],
            index=decomposed_dataset.values.index,
        )

    else:
        series = decomposed_dataset.values.observed

    if series.autocorr(lag=1) > 0.5:
        logging.info(f"Concluded that {decomposed_dataset.name} has a trend component")
    else:
        logging.info(
            f"Concluded that {decomposed_dataset.name} has a flat trend component"
        )

    return series.autocorr(lag=1) > 0.5


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


def __moving_average(data: Dataset, window_size: int = 12) -> pd.DataFrame:
    """Calculates the moving average of the data"""
    return data.rolling(window=window_size).mean()


def __extract_trend_equation_parameters(
    trend_component: pd.DataFrame,
) -> tuple:
    """Extracts the parameters from the trend equation"""
    trend_component = trend_component.dropna()
    x = np.arange(len(trend_component))
    y = trend_component.values
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


def __calculate_next_trend_values(
    train_trend_component: pd.DataFrame, number_of_steps: int
) -> pd.DataFrame:
    """Calculates the next trend values for the forecast"""
    logging.info(f"Calculating the next {number_of_steps} trend values")
    m, c = __extract_trend_equation_parameters(train_trend_component)
    x = np.arange(
        len(train_trend_component),
        len(train_trend_component) + number_of_steps,
    )
    return pd.DataFrame(m * x + c, index=x, columns=["trend"])


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


def __calculate_next_additive_seasonal_values(
    train_seasonal_component: pd.DataFrame,
    number_of_steps: int,
    seasonal_period: int,
) -> pd.Series:
    """Calculates the next seasonal values for the forecast assuming an additive model"""
    logging.info(f"Calculating the next {number_of_steps} seasonal values")
    seasonal_values_one_season = train_seasonal_component.values[-seasonal_period:]
    seasonal_values_test = []
    for i in range(number_of_steps):
        seasonal_values_test.append(seasonal_values_one_season[i % seasonal_period])
    print(f"len seasonal_values_test\n{len(seasonal_values_test)}")

    logging.info(
        f"Sample of the next {number_of_steps} seasonal values, assuming an additive model: \n{seasonal_values_test[:5]}"
    )

    return pd.Series(
        seasonal_values_test,
        index=range(
            len(train_seasonal_component),
            len(train_seasonal_component) + number_of_steps,
        ),
        name="seasonal",
    )


def __calculate_next_multiplicative_seasonal_values(
    train_seasonal_component: pd.DataFrame,
    number_of_steps: int,
    seasonal_period: int,
    multiplicative_factor: float,
) -> pd.Series:
    """Calculates the next seasonal values for the forecast assuming a multiplicative model"""
    logging.info(f"Calculating the next {number_of_steps} seasonal values")
    seasonal_values_one_season = train_seasonal_component.values[-seasonal_period:]
    seasonal_values_test = []
    for i in range(number_of_steps):
        seasonal_values_test.append(
            multiplicative_factor * seasonal_values_one_season[i % seasonal_period]
        )
    print(f"len seasonal_values_test\n{len(seasonal_values_test)}")
    logging.info(
        f"Sample of the next {number_of_steps} seasonal values, assuming a multiplicative model: \n{seasonal_values_test[:5]}"
    )

    return pd.Series(
        seasonal_values_test,
        index=range(
            len(train_seasonal_component),
            len(train_seasonal_component) + number_of_steps,
        ),
        name="seasonal",
    )


def __detect_an_increase_in_a_series(series: pd.Series) -> Tuple[bool, float]:
    """Detects if there is a gradual increase or decrease in a series
    by calculating the average of the first and last 20% of the series
    and comparing them.
    If the difference is significant, then there is a trend."""
    logging.info(f"Detecting trend in {series.name}")
    series = series.dropna()
    # if series is too short, then we cannot detect a trend
    if len(series) < 10:
        return False, 0

    first_20_percent = series[: int(len(series) * 0.2)].mean()

    # to avoid dvision by zero errors
    if first_20_percent == 0:
        first_20_percent += 0.01

    last_20_percent = series[-int(len(series) * 0.2) :].mean()
    logging.info(f"{first_20_percent} {last_20_percent}")
    logging.info(f"absolutediff: {abs(first_20_percent - last_20_percent)}")
    increase_or_decrease_present = abs(first_20_percent - last_20_percent) > (
        0.1 * series.mean()
    )

    # calcaulte the percentage increase or decrease in the magnitude of the series
    average_multiplier = last_20_percent / first_20_percent

    average_multiplier_per_step = average_multiplier ** (1 / len(series))

    return increase_or_decrease_present, average_multiplier_per_step


def __determine_if_seasonality_is_multiplicative(
    training_dataset: Dataset,
) -> Tuple[bool, float]:
    """Determines if the seasonal component is multiplicative"""
    logging.info("Determining if seasonality is multiplicative")
    if type(training_dataset.values) is pd.DataFrame:
        series = pd.Series(
            training_dataset.values[training_dataset.subset_column_name],
            index=training_dataset.values.index,
        )
    else:
        series = training_dataset.values.observed
    seasonal_period = __get_seasonal_period(training_dataset)
    logging.info(f"Seasonal period: {seasonal_period}")
    seasonal_index = series / __moving_average(series, seasonal_period)
    data_minus_moving_av_index = series - __moving_average(series, seasonal_period)
    seasonal_index.dropna(inplace=True)
    data_minus_moving_av_index.dropna(inplace=True)

    rolling_max = data_minus_moving_av_index.rolling(window=seasonal_period).max()
    logging.info(
        f"rolling max for {training_dataset.name} every 20: {rolling_max[::20]}"
    )

    increase_decrease_present, multiplicative_factor = __detect_an_increase_in_a_series(
        rolling_max
    )

    logging.info(f"Multiplicative factor: {multiplicative_factor}")

    return increase_decrease_present, multiplicative_factor



def __seasonal_decompose_data(data: Dataset) -> Dataset:
    """First gets the training data, then decomposes the data into trend, 
    seasonal and residual components.
    Return the full decomposition object inside the Dataset object"""

    training_dataset = __get_training_set(data)

    logging.info(f"Decomposing {data.name} with STL")

    return seasonal_decompose_data(training_dataset)



def __fit_model(stl_decomposition_dataset: Dataset) -> Model:
    """Fits the SES model to the data"""
    logging.info(f"Fitting SES model to {stl_decomposition_dataset.name}")

    de_trended_residual = np.abs(stl_decomposition_dataset.values.resid)  # remove negative values

    return SimpleExpSmoothing(endog=de_trended_residual).fit(
        smoothing_level=__alpha,
        optimized=False,
    )


def __sum_of_two_series(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Adds two series together and ensures the index is correct"""

    series2.index = series1.index
    return series1.add(series2, fill_value=0)


def __product_of_two_series(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Multiplies two series together and ensures the index is correct"""

    series2.index = series1.index
    return series1.mul(series2, fill_value=1)


def __predict(
    model: Model, decomposed_dataset: Dataset, data: Dataset
) -> PredictionData:
    """Predicts the next values in the time series"""
    title = f"{decomposed_dataset.subset_column_name} forecast for {decomposed_dataset.subset_row_name} with SES"
    forecasted_resid = model.forecast(
        __number_of_steps(data)
    )  # out of sample prediction
    forecasted_resid_in_sample = model.predict(
        start=1, end=len(decomposed_dataset.values.resid)
    )  # in sample prediction
    # add seasonal and trend components back to the forecast for the correct number of steps
    # if __determine_if_trend_with_acf(decomposed_dataset):
    logging.debug(f"Trend component present for {decomposed_dataset.name} with SES")
    trend_component = __calculate_next_trend_values(
        decomposed_dataset.values.trend, __number_of_steps(data)
    )
    trend_component_in_sample = decomposed_dataset.values.trend

    logging.debug(f"Calculating residual values for {decomposed_dataset.name}")
    forecasted_resid = __sum_of_two_series(forecasted_resid, trend_component["trend"])
    forecasted_resid_in_sample = __sum_of_two_series(
        forecasted_resid_in_sample, trend_component_in_sample
    )

    if __determine_if_seasonal_with_acf(decomposed_dataset):
        logging.debug(
            f"Seasonal component present for {decomposed_dataset.name} with SES... calculating seasonal values"
        )

        is_multiplicative, factor = __determine_if_seasonality_is_multiplicative(
            decomposed_dataset
        )

        if is_multiplicative:
            logging.debug(
                f"Seasonality is multiplicative for {decomposed_dataset.name}"
            )
            seasonal_component = __calculate_next_multiplicative_seasonal_values(
                decomposed_dataset.values.seasonal,
                __number_of_steps(data),
                __get_seasonal_period(data),
                factor,
            )
        else:
            logging.debug(f"Seasonality is additive for {decomposed_dataset.name}")
            seasonal_component = __calculate_next_additive_seasonal_values(
                decomposed_dataset.values.seasonal,
                __number_of_steps(data),
                __get_seasonal_period(data),
            )

        seasonal_component_in_sample = decomposed_dataset.values.seasonal
        # TODO: remove this once have solved the gap in the plot between the in sample and out of sample
        print(
            f"\n\nseasonal_component_in_sample tail: {seasonal_component_in_sample.tail()}"
        )
        print(f"\n\nnext seasonal_component head: {seasonal_component.head()}")
        forecasted_resid = __sum_of_two_series(forecasted_resid, seasonal_component)
        forecasted_resid_in_sample = __sum_of_two_series(
            forecasted_resid_in_sample, seasonal_component_in_sample
        )
        # TODO: remove this once have solved the gap in the plot between the in sample and out of sample
        print(
            f"\n\nforecasted_resid_in_sample tail: {forecasted_resid_in_sample.tail()}"
        )
        print(f"forecasted_resid head: {forecasted_resid.head()}")
    return PredictionData(
        method_name="SES",
        values=forecasted_resid,
        prediction_column_name=None,
        ground_truth_values=__get_test_set(data).values[data.subset_column_name],
        confidence_columns=None,
        title=title,
        plot_folder=f"{data.name}/{data.subset_row_name}/SES/",
        plot_file_name=f"{data.subset_column_name}_forecast",
        confidence_on_mean=False,
        confidence_method=None,
        color=get_color_map_by_method("SES"),
        in_sample_prediction=forecasted_resid_in_sample,
    )


ses = method(__seasonal_decompose_data, __fit_model, __predict)
