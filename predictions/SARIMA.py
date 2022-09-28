""" The SARIMA model

The SARIMA model is an extension of the ARIMA model. SARIMA stands for Seasonal 
AutoRegressive Integrated Moving Average.
The SARIMA model is used to model time series data with a seasonal component. 
The seasonal component is modeled using an ARIMA model whose order is differenced 
by the period of the seasonality.

SARIMA models are denoted SARIMA(p, d, q)(P, D, Q)s where the parameters are as 
follows:
The first three are the same as the ARIMA model and model the trend component:
- p: The number of lag observations included in the model, also 
called the trend autoregression order.
- d: The number of times that the raw observations are differenced, also 
called the degree of differencing, or the trend difference order.
- q: The size of the moving average window, also called the trend order 
of moving average.

The reaminder are not part of the ARIMA model and must be configured to
model the seasonal component:
- P: The seasonal autoregressive order.
- D: The seasonal difference order.
- Q: The seasonal moving average order.
- s: The number of time steps for a single seasonal period.

Importantly, the 's' parameter influences the P, D, Q parameters. 
For example, if s = 12 for monthly data, then a P=1 would make use of 
the first 12 lags of the seasonal difference (t-12), a P=2 would make 
use of the first 24 lags of the seasonal difference (t-12, t-24), and so on.

Similarly for D and Q, a D=1 would make use of the first 12 lags of 
the seasonal difference (t-12), a D=2 would make use of the first 24
lags of the seasonal difference (t-12, t-24), and so on. 
A Q =1 would make use of the first 12 lags of the seasonal moving 
average (t-12), a Q=2 would make use of the first 24 lags of the error
(t-12, t-24), and so on.

The trend elements (p, d, q) can be chosen through careful examination 
of the ACF and PACF plots, looking at correlations of recent lags.

The best Python library for SARIMA models is the statsmodels library. 
The statsmodels library provides the SARIMAX class that can be used to fit
SARIMA models. The implementation is called SARIMAX because it can also 
model exogenous variables. Hence the 'X' in SARIMAX.

The SARIMAX class has a fit() method that can be used to fit the model.

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
    print(f"differencing term d for {data.name} is {ndiffs(data.values[data.subset_column_name], test='adf')}")
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
    print(f"differencing term D for {data.name} is {ndiffs(data.values[data.subset_column_name], test='adf', max_d=maximum_differencing_term)}")
    return ndiffs(data.values[data.subset_column_name], test="adf", max_d=maximum_differencing_term)



def __fit_model(data: Dataset) -> Model:
    """Fit the SARIMA model to the first 80% of the data"""
    if __stationarity(data):
        logging.info(f"Data {data.name} is stationary; no differencing required")
        d = 0
    else:
        logging.info(f"Data {data.name} is not stationary")
        d = __get_differencing_term(data)

    D = __get_seasonal_differencing_term(data)
    s = __get_seasonal_period(data)

    my_order = (1, d, 1)
    my_seasonal_order = (1, D, 1, s)

    model = SARIMAX(
        __get_training_set(data),
        order=my_order,
        seasonal_order=my_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    print(f"Number of observations for {data.name} is {len(data.values)}")
    print(f"Training SARIMA model on {data.name} with order {my_order} and seasonal order {my_seasonal_order}")
    if data.time_unit == "days":
        # show progress bar for model fitting
        return model.fit(disp=True)
    else:
        return model.fit(disp=False)


def __forecast(model: Model, data: Dataset) -> pd.DataFrame:
    """Forecast the next 20% of the data"""
    title = f"{data.subset_column_name} forecast for {data.subset_row_name} for the next {__number_of_steps(data)} {data.time_unit} with SARIMA"
    print(f"Forecasting {title}")
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
