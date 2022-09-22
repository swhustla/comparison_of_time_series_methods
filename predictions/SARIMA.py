""" The SARIMA model

The SARIMA model is an extension of the ARIMA model. SARIMA stands for Seasonal 
AutoRegressive Integrated Moving Average.
The SARIMA model is used to model time series data with a seasonal component. 
The seasonal component is modeled using an ARIMA model whose order is differenced 
by the period of the seasonality.

SARIMA models are denoted SARIMA(p, d, q)(P, D, Q)s where the parameters are as 
follows:
- p: The number of lag observations included in the model, also called the lag order.
- d: The number of times that the raw observations are differenced, also called the
degree of differencing.
- q: The size of the moving average window, also called the order of moving average.
- P: The seasonal autoregressive order.
- D: The seasonal difference order.
- Q: The seasonal moving average order.
- s: The number of time steps for a single seasonal period.

The best Python library for SARIMA models is the statsmodels library. 
The statsmodels library provides the SARIMAX class that can be used to fit
SARIMA models.

SARIMA assumes that the data is stationary. If the data is not stationary, the
differencing term d is set to the number of differencing required to make the data
stationary. The differencing term D is set to the number of seasonal differencing
required to make the data stationary.

"""

from typing import TypeVar
from methods.SARIMA import sarima as method
import pandas as pd

import logging

from arch.unitroot import ADF
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.statespace.sarimax import SARIMAX

from data.Data import Dataset, Result
from predictions.Prediction import PredictionData

Model = TypeVar("Model")


def __number_of_steps(data: Dataset) -> int:
    return int(len(data.values) // 5)


def __get_training_set(data: Dataset) -> pd.DataFrame:
    return data.values[: -__number_of_steps(data)][data.subset_column_name]


def __get_test_set(data: Dataset) -> pd.DataFrame:
    return data.values[-__number_of_steps(data) :][data.subset_column_name]


def __stationarity(data: Dataset) -> bool:
    """Determines if the data is stationary"""
    data_to_check = data.values[data.subset_column_name]
    return ADF(data_to_check).pvalue < 0.05


def __get_differencing_term(data: Dataset) -> int:
    """Get the differencing term for the SARIMA model"""
    return ndiffs(data.values[data.subset_column_name], test="adf")


def __get_seasonal_period(data: Dataset) -> int:
    """Get the seasonal period for the SARIMA model"""
    if data.time_unit == "days":
        return 365
    elif data.time_unit == "weeks":
        return 52
    elif data.time_unit == "months":
        return 12
    else:
        raise ValueError("Invalid time unit")


def __get_seasonal_differencing_term(data: Dataset) -> int:
    """Get the seasonal differencing term for the SARIMA model"""
    maximum_differencing_term = 2
    return ndiffs(data.values[data.subset_column_name], test="adf", max_d=maximum_differencing_term)



def __fit_model(data: Dataset) -> Model:
    """Fit the SARIMA model to the first 80% of the data"""
    if __stationarity(data):
        logging.info(f"Data {data.name} is stationary; no differencing required")
        d = 0
    else:
        logging.info(f"Data {data.name} is not stationary")
        logging.info(f"Differencing term is {__get_differencing_term(data)}")
        d = __get_differencing_term(data)

    D = __get_seasonal_differencing_term(data)
    s = __get_seasonal_period(data)

    model = SARIMAX(
        __get_training_set(data),
        order=(1, d, 1),
        seasonal_order=(1, D, 1, s),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def __forecast(model: Model, data: Dataset) -> pd.DataFrame:
    """Forecast the next 20% of the data"""
    title = f"{data.subset_column_name} forecast for {data.subset_row_name} for the next {__number_of_steps(data)} {data.time_unit} with SARIMA"
    return PredictionData(
        values=model.get_forecast(
            steps=__number_of_steps(data)
        ).predicted_mean,
        prediction_column_name=None,
        ground_truth_values=__get_test_set(data),
        confidence_columns=None,
        title=title,
    )


sarima = method(__fit_model, __forecast)
