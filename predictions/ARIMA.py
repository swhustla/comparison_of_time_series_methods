"""
The ARIMA method

This method uses the ARIMA model to predict the next 20% of the data.
ARIMA stands for AutoRegressive Integrated Moving Average. It was created by Box and Jenkins 
in 1970, building on techniques used by statisticians to forecast economic data.

It was shown to be a very effective model for time series data, in particular on financial data.
Accurate forecasting is important in finance, as it allows investors to make better decisions.

Downsides to the ARIMA model are that it is not very flexible, and it is not very good at
forecasting long term trends. It is also not very good at forecasting data with a seasonal 
component, such as weather data.

The ARIMA model is a combination of three models:
- AutoRegressive (AR): A linear regression model that uses the dependent relationship between 
an observation and some number of lagged observations.
- Integrated (I): The use of differencing of raw observations in order to make the time series
stationary.
- Moving Average (MA): A model that uses the dependency between an observation and a residual
error from a moving average model applied to lagged observations.

The ARIMA model works by fitting a linear regression model to the data, and then using the 
residuals from that model to fit a moving average model. The coefficients from both of these
models are then used to predict the next values in the time series.

The training data is the first 80% of the dataset, and the model is then used to predict the 
next 20% of the data.

"""

from typing import TypeVar
from methods.ARIMA import arima as method
import pandas as pd

import logging

from arch.unitroot import ADF
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.forecasting.stl import STLForecast
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error


from data.dataset import Dataset, Result
from predictions.Prediction import PredictionData


import measurements.get_metrics as get_metrics

Model = TypeVar("Model")


def __number_of_steps(data: Dataset) -> int:
    return int(len(data.values) // 5)


def __get_training_set(data: Dataset) -> pd.DataFrame:
    return data.values[: -__number_of_steps(data)][data.subset_column_name]


def __get_test_set(data: Dataset) -> pd.DataFrame:
    return data.values[-__number_of_steps(data) :][data.subset_column_name]


def __get_period_of_seasonality(data: Dataset) -> int:
    """
    Returns the period of seasonality.
    """
    if data.time_unit == "years":
        return 11
    elif data.time_unit == "months":
        return 12
    elif data.time_unit == "weeks":
        return 52
    elif data.time_unit == "days":
        return 365

def __get_seasonal_parameter(data: Dataset) -> int:
    """Get the seasonal parameter for the ARIMA model"""
    if data.time_unit == "days":
        return 7
    elif data.time_unit == "weeks":
        return 4
    elif data.time_unit == "months":
        return 12
    elif data.time_unit == "years":
        return 1


def __stationarity(data: Dataset) -> bool:
    """
    Checks if the data is stationary using the Augmented Dickey-Fuller test.
    A p-value of less than 0.05 indicates that the data is stationary.
    """
    data_to_check = data.values[data.subset_column_name]
    return ADF(data_to_check).pvalue < 0.05


def __get_differencing_term(data: Dataset) -> int:
    """Get the differencing term for the ARIMA model"""
    return ndiffs(data.values[data.subset_column_name], test="adf")

def __evaluate_arima_model(data: Dataset, arima_order: tuple) -> Result:
    """Evaluate an ARIMA model for a given order (p,d,q) and return RMSE"""
    # prepare training dataset
    training_set = __get_training_set(data)
    # prepare test dataset
    test_set = __get_test_set(data)
    # make predictions
    model = STLForecast(
        training_set,
        period=__get_period_of_seasonality(data),
        model=sm.tsa.ARIMA,
        model_kwargs={"order": arima_order},
    )
    model_fit = model.fit()
    yhat = model_fit.forecast(len(test_set))
    # calculate out of sample error 
    mse = mean_squared_error(test_set, yhat)
    return mse


def __evaluate_models(data: Dataset, p_values: list, d_values: list, q_values: list) -> Result:
    """Evaluate combinations of p, d and q values for an ARIMA model"""
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = __evaluate_arima_model(data, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    logging.info(f"ARIMA{order} MSE={mse}")
                except Exception as e:
                    logging.error(f"ARIMA{order} failed with error: {e}")
                    continue
    logging.info(f"Best ARIMA{best_cfg} MSE={best_score}")
    return best_cfg

    
__p_values = [0, 1, 2, 4, 6, 8, 10]
__d_values = [0, 1, 2]
__q_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def __get_best_model_order(data: Dataset) -> Model:
    """Get the best model order for the ARIMA model"""
    return __evaluate_models(data, __p_values, __d_values, __q_values)

def __fit_auto_regressive_model(data: Dataset) -> Model:
    """Fit the best ARIMA model to the first 80% of the data"""

    ar_order, differencing_term, ma_order = __get_best_model_order(data)

    model = STLForecast(
        __get_training_set(data),
        period=__get_period_of_seasonality(data),
        model=sm.tsa.ARIMA,
        model_kwargs={"order": (ar_order, differencing_term, ma_order)},
    )

    model_result = model.fit().model_result
    return model_result


def __get_model_order_snake_case(model: Model) -> str:
    """convert model order dict to snake case filename"""

    model_order = model.model.order
    model_order = f"AR{model_order[0]}_I{model_order[1]}_MA{model_order[2]}"
    return model_order.replace(" ", "_")


def __forecast(model: Model, data: Dataset) -> PredictionData:
    """Forecast the next 20% of the data"""
    title = f"{data.subset_column_name} forecast for {data.subset_row_name} with ARIMA"
    # print(f"Sample of the forecast for {data.name}: \n{model.get_forecast(__number_of_steps(data)).summary_frame()}")
    return PredictionData(
        values=model.get_forecast(steps=__number_of_steps(data)).summary_frame(),
        prediction_column_name="mean",
        ground_truth_values=__get_test_set(data),
        confidence_columns=["mean_ci_lower", "mean_ci_upper"],
        title=title,
        plot_folder=f"{data.name}/{data.subset_row_name}/ARIMA/",
        plot_file_name=f"{data.subset_column_name}_forecast_{__get_model_order_snake_case(model)}",
    )


arima = method(__fit_auto_regressive_model, __forecast)
