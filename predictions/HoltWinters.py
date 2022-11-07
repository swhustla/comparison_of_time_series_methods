"""
Holt-Winters

This technique is a time series forecasting method for univariate data.
It is a triple exponential smoothing technique that takes into account
trend and seasonality.

It was created by Charles C. Holt and Peter Winters in 1957, and was
used to forecast the demand for coal in the UK. It was accurate enough
to be used in the UK coal industry for 20 years. It was also used to
forecast the demand for electricity in the UK later, in 1970.

The Holt-Winters method is a generalization of the Holt method, which
only takes into account trend.

It contains two main components:
- The level component (L): The average value of the data over time.
- The trend component (T): The average change in the data over time.

The Holt-Winters method also contains a seasonal component (S), which
is the average change in the data over a seasonal period.

The Holt-Winters method is a good choice for forecasting data with a
strong seasonal component, such as weather data.

The starting components are calculated using the following equations:
- L0 = (y1 + y2 + ... + ym) / m
- T0 = (y2 - y1 + y3 - y2 + ... + ym - ym-1) / (m - 1)

where m is the number of data points in a seasonal period.

The first step is then:
- L1 = y1
- T1 = T0

For the next steps:
- L(t) = alpha * y(t) + (1 - alpha) * (L(t-1) + T(t-1))
- T(t) = beta * (L(t) - L(t-1)) + (1 - beta) * T(t-1)

where alpha and beta are smoothing parameters.

The seasonal component is then calculated using the following equation:
- S(t) = gamma * (y(t) - L(t-1) - T(t-1)) + (1 - gamma) * S(t-m)

where gamma is a smoothing parameter.

The forecast is then calculated using the following equation:
- y(t+h) = L(t) + h * T(t) + S(t - m + (h - 1) % m)

where h is the number of steps to forecast.

The Holt-Winters method is a good baseline to compare other methods
against.

The best Python library for Holt-Winters models is statsmodels. It has
a holtwinters module that provides a HoltWintersResults class that
implements the Holt-Winters method.

In this implementation  we will use a grid search to find the best set
of parameters for the Holt-Winters model. We will use the mean absolute
percentage error (MAPE) as the metric to evaluate the model.

The grid search will try to tune the following parameters:
- trend: Whether to include a trend component.
- damped_trend: Whether to include a damped trend component.
- seasonal: Whether to include a seasonal component.
- seasonal_periods: The number of data points in a seasonal period.
- use_boxcox: Whether to use the Box-Cox transform, or to use the
    log transform if the data is negative.

The grid search will try all combinations of the parameters, and will
return the best set of parameters.



"""


from typing import Tuple, TypeVar, Callable, List, Dict, Any

from typing import TypeVar, Callable
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from joblib import Parallel, delayed

import pandas as pd
import numpy as np

import logging

from data.dataset import Dataset, Result
from predictions.Prediction import PredictionData

from methods.HoltWinters import holt_winters as method

from methods.get_metrics import get_metrics


Model = TypeVar("Model")


def __number_of_steps(data: Dataset) -> int:
    return int(len(data.values) // 5)


def __get_training_set(data: Dataset) -> Dataset:
    return Dataset(
        name=data.name,
        values=data.values[: -__number_of_steps(data)][data.subset_column_name],
        subset_column_name=data.subset_column_name,
        number_columns=data.number_columns,
        time_unit=data.time_unit,
        subset_row_name=data.subset_row_name,
        seasonality=data.seasonality,
    )


def __get_test_set(data: Dataset) -> Dataset:
    return Dataset(
        name=data.name,
        values=data.values[-__number_of_steps(data) :][data.subset_column_name],
        subset_column_name=data.subset_column_name,
        number_columns=data.number_columns,
        time_unit=data.time_unit,
        subset_row_name=data.subset_row_name,
        seasonality=data.seasonality,
    )


def __exp_smoothing_forecast(data: Dataset, config: dict) -> np.array:
    # multi-step Holt-Winters Exponential Smoothing forecast
    t, d, s, p, b, r = config
    # define model
    training_data = np.array(__get_training_set(data).values)
    model = ExponentialSmoothing(
        training_data,
        trend=t,
        damped_trend=d,
        seasonal=s,
        seasonal_periods=p,
        use_boxcox=b,
    )

    # fit model
    model_fit = model.fit(optimized=True, remove_bias=r)
    # make multi-step forecast
    yhat = model_fit.predict(
        len(training_data), len(training_data) + __number_of_steps(data) - 1
    )
    return yhat


def __measure_rmse(actual: Dataset, predicted: np.array) -> float:
    return np.sqrt(np.mean((actual.values - predicted) ** 2))


def __measure_mape(actual: Dataset, predicted: np.array) -> float:
    return np.mean(np.abs((actual.values - predicted) / actual.values)) * 100


def __validation(data: Dataset, config: dict) -> float:
    """Validation for uni-variate data."""
    test = __get_test_set(data)
    yhat = __exp_smoothing_forecast(data, config)
    # estimate prediction error
    error = __measure_mape(test, yhat)

    return error


def __score_model(
    data: Dataset,
    config: dict,
    func: Callable[[Dataset, dict], float],
    debug: bool = False,
) -> Tuple[float, dict]:
    """Score a model, return None on failure."""
    result = None
    # convert config to a key
    key = str(config)
    # show all warnings and fail on exception if debugging
    if debug:
        result = func(data, config)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                result = func(data, config)
        except Exception as e:
            logging.error(e)
            result = None
    # check for an interesting result
    if result is not None:
        print(f" > Model[{key}] {result}")
    return (result, config)


def __grid_search_configs(
    data: Dataset, cfg_list: list, parallel: bool = True
) -> List[Tuple[float, dict]]:
    """Grid search the configs."""
    scores = list()
    # use fork to avoid memory issues with multiprocessing
    if parallel:
        scores = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(__score_model)(data, cfg, __validation) for cfg in cfg_list
        )
    else:
        scores = [__score_model(data, cfg, __validation) for cfg in cfg_list]

    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[0])

    print(f"Printing top 3 config scores")
    for cfg, error in scores[:3]:
        print(f" > {cfg}, error = {error}")

    # remove empty results
    scores = [r for r in scores if r[0] is not None]

    return scores


def __exp_smoothing_configs(seasonal: List[int]) -> list:
    """Create a set of exponential smoothing configs to try."""
    model_configs = list()
    # define config lists
    t_params = ["add", "mul", None]
    d_params = [True, False]
    s_params = ["add", "mul", None]
    p_params = seasonal
    b_params = [True, False]
    r_params = [True, False]
    # create config instances
    null_trend_done = False
    null_seasonal_done = False
    for t in t_params:
        for d in d_params:
            if t is None and d:
                d = False
                null_trend_done = True
            if t is None and null_trend_done:
                continue
            for s in s_params:
                for p in p_params:
                    if s is None and p:
                        p = None
                        null_seasonal_done = True
                    if s is None and null_seasonal_done:
                        continue
                    for b in b_params:
                        for r in r_params:
                            cfg = [t, d, s, p, b, r]
                            model_configs.append(cfg)
    return model_configs


def __get_seasonal_period_list(data: Dataset) -> List[int]:
    """Get a list of seasonal periods to try.
    In Exponential Smoothing, the seasonal_periods parameter is the number of
    periods in each season. For example, if the data is monthly, and you believe
    there is a yearly seasonality, then the seasonal_periods parameter would be
    12. 
    """
    if data.seasonality is None:
        return [None]
    else:
        if data.time_unit == "days":
            return [365]
        elif data.time_unit == "weeks":
            return [4, 5, 10, 52]
        elif data.time_unit == "months":
            return [3, 12, 24]
        elif data.time_unit == "quarters":
            return [4, 5]
        elif data.time_unit == "years":
            return [11, 12, 13]
        else:
            return [None]


def __get_best_model(
    data: Dataset, parallel: bool = True
) -> Tuple[ExponentialSmoothing, dict]:
    """Get the best model for the data."""
    # define config lists
    cfg_list = __exp_smoothing_configs(seasonal=__get_seasonal_period_list(data))
    # grid search configs
    scores = __grid_search_configs(data, cfg_list, parallel)
    # get the best config
    best_cfg = scores[0][1]
    print(f"Best trend: {best_cfg[0]}")
    print(f"Best damped trend: {best_cfg[1]}")
    print(f"Best seasonal: {best_cfg[2]}")
    print(f"Best seasonal period: {best_cfg[3]}")
    print(f"Best boxcox: {best_cfg[4]}")
    print(f"Best remove bias: {best_cfg[5]}")

    # fit model to training data
    model = ExponentialSmoothing(
        __get_training_set(data).values,
        trend=best_cfg[0],
        damped_trend=best_cfg[1],
        seasonal=best_cfg[2],
        seasonal_periods=best_cfg[3],
        use_boxcox=best_cfg[4],
    )
    model_fit = model.fit(optimized=True, remove_bias=best_cfg[5])
    return model_fit, best_cfg


def __get_forecast(
    data: Dataset, model: ExponentialSmoothing, best_config: dict
) -> PredictionData:
    """Get the forecast for the data."""
    title = f"Forecast for {data.name} {data.subset_column_name} using Holt-Winters Exponential Smoothing"

    print(f"best config found for {data.name} {data.subset_column_name}: {best_config}")

    # make multi-step forecast
    yhat = model.predict(
        len(__get_training_set(data).values),
        len(__get_training_set(data).values) + __number_of_steps(data) - 1,
    )
    return PredictionData(
        values=yhat,
        prediction_column_name=None,
        ground_truth_values=__get_test_set(data).values,
        confidence_columns=None,
        title=title,
        plot_folder=f"{data.name}/{data.subset_row_name}/Holts-Winters-ES/",
        plot_file_name=f"{data.subset_column_name}_forecast",
        model_config=best_config,
    )


holt_winters = method(__get_best_model, __get_forecast)
