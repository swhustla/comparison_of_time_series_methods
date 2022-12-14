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

from typing import TypeVar, Generic, List, Tuple, Dict, Any, Optional, Union
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

__test_size: float = 0.2


def __number_of_steps(data: Dataset) -> int:
    return int(len(data.values) // 5)


def __get_training_set(data: Dataset) -> pd.DataFrame:
    return data.values[: -__number_of_steps(data)][data.subset_column_name]


def __get_test_set(data: Dataset) -> pd.DataFrame:
    return data.values[-__number_of_steps(data) :][data.subset_column_name]


def __get_train_and_test_sets(data: Dataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare the data for the ARIMA model"""
    training_set = __get_training_set(data)
    test_set = __get_test_set(data)
    return training_set, test_set


def __get_train_validation_and_test_sets(
    data: Dataset,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare the data for the ARIMA model"""
    training_set, test_set = __get_train_and_test_sets(data)
    validation_set = training_set[-__number_of_steps(data) :]
    training_set = training_set[: -__number_of_steps(data)]
    return training_set, validation_set, test_set


def __get_period_of_seasonality(data: Dataset) -> int:
    """
    Returns the period of seasonality.
    """
    if data.time_unit == "years":
        return 12
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


def __evaluate_arima_model(
    data: Dataset, arima_order: tuple, use_validation: bool, trend: str
) -> Result:
    """Evaluate an ARIMA model for a given order (p,d,q) and return RMSE"""
    # prepare training and test datasets
    if use_validation:
        training_set, validation_set, test_set = __get_train_validation_and_test_sets(
            data
        )
    else:
        training_set, test_set = __get_train_and_test_sets(data)
    # make predictions
    model = STLForecast(
        training_set,
        period=__get_period_of_seasonality(data),
        model=sm.tsa.ARIMA,
        model_kwargs={"order": arima_order, "trend": trend},
    )
    model_fit = model.fit()
    if use_validation:
        predictions = model_fit.forecast(len(validation_set))
        error = mean_squared_error(validation_set, predictions)
    else:
        predictions = model_fit.forecast(len(test_set))
        error = mean_squared_error(test_set, predictions)

    # return out of sample error
    return error


def __evaluate_models(
    data: Dataset, p_values: list, d_values: list, q_values: list, trend_values: list   
) -> Result:
    """Evaluate combinations of p, d and q values for an ARIMA model"""
    best_score, best_cfg, best_trend = float("inf"), None, None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for t in trend_values:
                    order = (p, d, q)
                    try:
                        mse = __evaluate_arima_model(
                            data, arima_order=order, use_validation=True, trend=t
                        )
                        if mse < best_score:
                            best_score, best_cfg, best_trend = mse, order, t
                        logging.info(f"ARIMA{order} Trend={t} MSE={mse}")
                    except Exception as e:
                        logging.error(f"ARIMA{order} Trend={t} failed with error: {e}")
                        continue
    logging.info(f"Best ARIMA: {best_cfg} Best trend: {best_trend} MSE={best_score}")
    return best_cfg, best_trend


__p_values = [0, 1, 2, 4, 8, 10]
__d_values = [0, 1, 2]
__q_values = [0, 1, 2, 4, 8, 10]
__trend_values = ["c", "t", "ct"]


def __calculate_number_of_configurations() -> int:
    return len(__p_values) * len(__d_values) * len(__q_values) * len(__trend_values)

def __get_best_model_order(data: Dataset) -> Model:
    """Get the best model order for the ARIMA model"""
    return __evaluate_models(data, __p_values, __d_values, __q_values, __trend_values)


def __fit_auto_regressive_model(data: Dataset) -> Model:
    """Fit the best ARIMA model to the first 80% of the data"""

    (ar_order, differencing_term, ma_order), trend = __get_best_model_order(data)

    model = STLForecast(
        endog=__get_training_set(data),
        period=__get_period_of_seasonality(data),
        model=sm.tsa.ARIMA,
        model_kwargs={"order": (ar_order, differencing_term, ma_order), "trend": trend},
    )

    return model.fit()


def __get_model_order_snake_case(model: Model) -> str:
    """convert model order dict to snake case filename"""

    model_order = model.model.order
    model_order = f"AR{model_order[0]}_I{model_order[1]}_MA{model_order[2]}"
    return model_order.replace(" ", "_")


def __forecast(model: Model, data: Dataset) -> PredictionData:
    """Forecast the next 20% of the data"""
    title = f"{data.subset_column_name} forecast for {data.subset_row_name} with ARIMA"
    prediction = model.forecast(__number_of_steps(data))
    length_in_sample = len(__get_training_set(data).values)
    prediction_in_sample = model.get_prediction(0,length_in_sample).summary_frame()
    prediction_summary = model.model_result.get_forecast(__number_of_steps(data)).summary_frame()
    combined_data = pd.concat([prediction, prediction_summary], axis=1)
    combined_data.rename(columns={0: "forecast"}, inplace=True)
    return PredictionData(
        method_name="ARIMA",
        values=combined_data,
        prediction_column_name="forecast",
        ground_truth_values=__get_test_set(data),
        confidence_columns=["mean_ci_lower", "mean_ci_upper"],
        title=title,
        plot_folder=f"{data.name}/{data.subset_row_name}/ARIMA/",
        plot_file_name=f"{data.subset_column_name}_forecast_{__get_model_order_snake_case(model)}",
        number_of_iterations=__calculate_number_of_configurations(),
        confidence_on_mean=True,
        confidence_method="95% confidence interval",
        color="orange",
        in_sample_prediction=prediction_in_sample.iloc[:, 0],
    )


arima = method(__fit_auto_regressive_model, __forecast)
