"""
The Auto Regressive Prediction method

Auto-regressive models are a class of statistical models for analyzing and forecasting time series data. They explicitly model the relationship between the observations and their lagged values. This is in contrast to other methods that model the relationship between the observations and a deterministic trend, such as linear or exponential trend.
They were first developed for analyzing and forecasting economic time series data, and are now widely used in other fields, such as signal processing and econometrics.

The AR model is defined by the following equation:

y_t = c + \sum_{i=1}^p \phi_i y_{t-i} + \epsilon_t

where y_t is the value of the time series at time t, c is a constant, 
\phi_i is the coefficient for lag i, and \epsilon_t is a white noise error term.

The AR model is a special case of the general linear regression model, where the regressors are lagged values of the dependent variable. The AR model is also a special case of the autoregressive moving average model, where the moving average coefficients are all zero.

It is equivalent to a moving average model of the same order, except that the moving average model uses the past forecast errors as regressors, while the AR model uses the past values of the dependent variable as regressors.

An ARIMA model can be used here with the MA and I components set to 0.

The best Python library for AR models is statsmodels. It provides a wide range of models, including AR, ARMA, ARIMA, ARIMAX, VAR, VARMAX, and SARIMAX. It also provides a wide range of tools for model selection, diagnostics, and visualization.


"""


from typing import TypeVar, Callable
from methods.AR import ar as method
import pandas as pd

import logging

from arch.unitroot import ADF
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.forecasting.stl import STLForecast
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf

import matplotlib.pyplot as plt

from data.Data import Dataset, Result
from predictions.Prediction import PredictionData

Model = TypeVar("Model")


def __number_of_steps(data: Dataset) -> int:
    return int(len(data.values) // 5)


def __get_training_set(data: Dataset) -> pd.DataFrame:
    return data.values[: -__number_of_steps(data)][data.subset_column_name]


def __get_test_set(data: Dataset) -> pd.DataFrame:
    return data.values[-__number_of_steps(data) :][data.subset_column_name]

def __quick_check_for_auto_correlation(data: Dataset) -> bool:
    """
    Checks if the data is auto-correlated using the Augmented Dickey-Fuller test.
    A p-value of less than 0.05 indicates that the data is auto-correlated.
    """
    adf = ADF(data.values[data.subset_column_name])
    return adf.pvalue < 0.05

def __get_number_of_lags(data: Dataset) -> int:
    """
    Returns the number of lags to use in the partial autocorrelation function.
    The number of lags is important for the model to be able to 
    capture the auto-correlation in the data.
    """
    if data.time_unit == "days":
        return 30
    elif data.time_unit == "weeks":
        return 51
    elif data.time_unit == "months":
        return 11
    else:
        return 1


def __transform_data(data: Dataset) -> pd.DataFrame:
    """
    Transforms the data to a pandas DataFrame.
    """
    return pd.DataFrame(data.values[data.subset_column_name])

def __determine_order_using_pacf(data: Dataset) -> int:
    """
    Determines the order of the model using the partial autocorrelation function.
    A plot of the partial autocorrelation function is shown.
    The user can then determine the order of the model by looking at the plot.
    """
    plot_pacf(__transform_data(data), lags=__get_number_of_lags(data), method="ywm")
    plt.show()
    return int(input("Enter the order of the model: "))


def __train_auto_regressive_model(data: Dataset) -> Model:
    """
    Trains an auto-regressive model using the training set.
    """
    ar_order = __determine_order_using_pacf(data)
    ma_order = 0
    int_order = 0

    model = STLForecast(
        __get_training_set(data),
        sm.tsa.ARIMA,
        model_kwargs=dict(order=(ar_order, int_order, ma_order), trend="t"),
    )
    model_result = model.fit().model_result
    return model_result

def __forecast(model: Model, data: Dataset) -> pd.DataFrame:
    """
    Makes a forecast using the trained model.
    """
    title =f"{data.subset_column_name} for {data.subset_row_name} forecast using Auto Regressive model"
    return PredictionData(
        values=model.get_forecast(steps=__number_of_steps(data)).summary_frame(),
        prediction_column_name="mean",
        ground_truth_values=__get_test_set(data),
        confidence_columns=["mean_ci_lower", "mean_ci_upper"],
        title=title,
    )

ar = method(
    __quick_check_for_auto_correlation,
    __train_auto_regressive_model,
    __forecast,
)