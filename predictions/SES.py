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

from typing import TypeVar
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

import pandas as pd
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)

from data.Data import Dataset, Result
from predictions.Prediction import PredictionData

from methods.SES import ses as method

Model = TypeVar("Model")

__alpha = 0.2


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
    if type(decomposed_dataset.values) is pd.DataFrame:
        series = pd.Series(
            decomposed_dataset.values[decomposed_dataset.subset_column_name],
            index=decomposed_dataset.values.index,
        )

    else:
        series = decomposed_dataset.values.observed
    return series.autocorr(lag=1) > 0.5


def __determine_if_seasonal_with_acf(dataset: Dataset) -> bool:
    """Determines if the data has a seasonal component"""
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
    else:
        return 1


def __calculate_next_seasonal_values(
    train_seasonal_component: pd.DataFrame,
    number_of_steps: int,
    seasonal_period: int,
) -> list:
    """Calculates the next seasonal values for the forecast"""
    seasonal_values_one_season = train_seasonal_component.values[
        -seasonal_period:
    ]
    seasonal_values_test = []
    for i in range(number_of_steps):
        seasonal_values_test.append(
            seasonal_values_one_season[i % seasonal_period]
        )

    return pd.Series(
        seasonal_values_test,
        index=range(
            len(train_seasonal_component),
            len(train_seasonal_component) + number_of_steps,
        ),
        name="seasonal",
    )


def __detect_an_increase_in_a_series(series: pd.Series) -> bool:
    """Detects if there is a gradual increase or decrease in a series"""
    return np.abs(series.diff().mean()) > 0.05


def __determine_if_seasonality_is_multiplicative(
    training_dataset: Dataset,
) -> bool:
    """Determines if the seasonal component is multiplicative"""
    if type(training_dataset.values) is pd.DataFrame:
        series = pd.Series(
            training_dataset.values[training_dataset.subset_column_name],
            index=training_dataset.values.index,
        )
    else:
        series = training_dataset.values.observed
    seasonal_period = __get_seasonal_period(training_dataset)
    seasonal_index = series / __moving_average(series, seasonal_period)
    data_minus_moving_av_index = series - __moving_average(
        series, seasonal_period
    )
    seasonal_index.dropna(inplace=True)
    data_minus_moving_av_index.dropna(inplace=True)

    rolling_max = data_minus_moving_av_index.rolling(
        window=seasonal_period
    ).max()
    return __detect_an_increase_in_a_series(rolling_max)


def __seasonal_decompose_data(data: Dataset) -> Dataset:
    """Removes trend and seasonality from the training data"""

    training_data = __get_training_set(data)

    seasonal_component_present = __determine_if_seasonal_with_acf(
        training_data
    )

    seasonal_period = __get_seasonal_period(training_data)

    if seasonal_component_present:
        seasonal_component = (
            "multiplicative"
            if __determine_if_seasonality_is_multiplicative(training_data)
            else "additive"
        )
        logging.info(
            f"Seasonal component type: {seasonal_component} for {training_data.name}"
        )

        decomposition = seasonal_decompose(
            training_data.values[training_data.subset_column_name],
            model=seasonal_component,
            period=seasonal_period,
            extrapolate_trend="freq",
        )
    else:
        seasonal_component = "additive"

        decomposition = seasonal_decompose(
            training_data.values[training_data.subset_column_name],
            model=seasonal_component,
            period=seasonal_period,
            extrapolate_trend="freq",
        )

    return Dataset(
        name=data.name,
        time_unit=data.time_unit,
        number_columns=data.number_columns,
        values=decomposition,
        subset_column_name=data.subset_column_name,
        subset_row_name=data.subset_row_name,
        seasonality=data.seasonality,
    )


def __tidy_up_decomposition_data(data: Dataset) -> Dataset:
    """Tidies up the decomposition data"""

    residual_values = data.values.resid
    residual_values = np.abs(residual_values)  # remove negative values

    return Dataset(
        name=data.name,
        time_unit=data.time_unit,
        number_columns=data.number_columns,
        values=residual_values,
        subset_column_name=data.subset_column_name,
        subset_row_name=data.subset_row_name,
        seasonality=data.seasonality,
    )


def __fit_model(data: Dataset) -> Model:
    """Fits the model to the data"""
    de_trended_data = __tidy_up_decomposition_data(data).values

    return SimpleExpSmoothing(de_trended_data).fit(
        smoothing_level=__alpha,
        optimized=False,
    )


def __sum_of_two_series(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Adds two series together and ensures the index is correct"""

    series2.index = series1.index
    return series1.add(series2, fill_value=0)


def __product_of_two_series(
    series1: pd.Series, series2: pd.Series
) -> pd.Series:
    """Multiplies two series together and ensures the index is correct"""

    series2.index = series1.index
    return series1.mul(series2, fill_value=1)


def __predict(
    model: Model, decomposed_dataset: Dataset, data: Dataset
) -> PredictionData:
    """Predicts the next values in the time series"""
    title = f"{decomposed_dataset.subset_column_name} forecast for {decomposed_dataset.subset_row_name} with SES"
    forecasted_resid = model.forecast(__number_of_steps(data))

    # add seasonal and trend components back to the forecast for the correct number of steps
    if __determine_if_trend_with_acf(decomposed_dataset):
        trend_component = __calculate_next_trend_values(
            decomposed_dataset.values.trend, __number_of_steps(data)
        )

        forecasted_resid = __sum_of_two_series(
            forecasted_resid, trend_component["trend"]
        )

    if __determine_if_seasonal_with_acf(decomposed_dataset):
        seasonal_component = __calculate_next_seasonal_values(
            decomposed_dataset.values.seasonal,
            __number_of_steps(data),
            __get_seasonal_period(data),
        )
        if __determine_if_seasonality_is_multiplicative(decomposed_dataset):
            seasonal_component_type = "multiplicative"
        else:
            seasonal_component_type = "additive"

        logging.info(
            f"Type of seasonal component: {seasonal_component_type} for {data.name}"
        )
        if seasonal_component_type == "multiplicative":
            forecasted_resid = __product_of_two_series(
                forecasted_resid, seasonal_component
            )
        else:
            forecasted_resid = __sum_of_two_series(
                forecasted_resid, seasonal_component
            )

    return PredictionData(
        values=forecasted_resid,
        prediction_column_name=None,
        ground_truth_values=__get_test_set(data).values[
            data.subset_column_name
        ],
        confidence_columns=None,
        title=title,
        plot_folder=f"{data.name}/{data.subset_row_name}/SES/",
        plot_file_name=f"{data.subset_column_name}_forecast",
    )


ses = method(__seasonal_decompose_data, __fit_model, __predict)
