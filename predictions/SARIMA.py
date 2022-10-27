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
called the trend autoregression order, or "AR order".
- d: The number of times that the raw observations are differenced, also 
called the degree of differencing, or the trend difference order; "trend order".
- q: The size of the moving average window, also called the trend order 
of moving average; "MA order".

The reaminder are not part of the ARIMA model and must be configured to
model the seasonal component:
- P: The seasonal autoregressive order, or "seasonal AR order".
- D: The seasonal difference order, or "seasonal difference order".
- Q: The seasonal moving average order, or "seasonal MA order".
- s: The number of time steps for a single seasonal period; "seasonal period".

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

from typing import TypeVar, Generic, List, Tuple, Callable, Dict, Any
from methods.SARIMA import sarima as method
import pandas as pd
import numpy as np

from multiprocessing import cpu_count
from joblib import Parallel, delayed
from warnings import catch_warnings, filterwarnings

import logging

from arch.unitroot import ADF
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_error

from data.dataset import Dataset, Result
from predictions.Prediction import PredictionData

Model = TypeVar("Model")

__best_score = float("inf")


def __number_of_steps(data: Dataset) -> int:
    return int(len(data.values) // 5)


def __get_training_set(data: Dataset) -> pd.DataFrame:
    return data.values[: -__number_of_steps(data)][data.subset_column_name]


def __get_test_set(data: Dataset) -> pd.DataFrame:
    return data.values[-__number_of_steps(data) :][data.subset_column_name]


def __get_training_and_test_sets(data: Dataset) -> tuple:
    """Get the training and test sets for the SARIMA model"""
    training_set = __get_training_set(data)
    test_set = __get_test_set(data)
    return training_set, test_set


def __get_training_validation_and_test_sets(data: Dataset) -> tuple:
    """Get the training, validation and test sets for the SARIMA model"""
    training_set, test_set = __get_training_and_test_sets(data)
    validation_size = int(len(training_set) * 0.2)
    validation_set = training_set[-validation_size:]
    training_set = training_set[:-validation_size]
    # logging.info (f"training set size for {data.name} is {len(training_set)}")
    return training_set, validation_set, test_set


def __stationarity(data: Dataset) -> bool:
    """Determines if the data is stationary"""
    data_to_check = data.values[data.subset_column_name]
    return ADF(data_to_check).pvalue < 0.05


def __get_differencing_term(data: Dataset) -> int:
    """Get the differencing term for the SARIMA model"""
    logging.info(
        f"differencing term d for {data.name} is {ndiffs(data.values[data.subset_column_name], test='adf')}"
    )
    return ndiffs(data.values[data.subset_column_name], test="adf")


def __get_seasonal_period(data: Dataset) -> int:
    """Get the seasonal period for the SARIMA model"""
    if data.time_unit == "days":
        return 365
    elif data.time_unit == "weeks":
        return 52
    elif data.time_unit == "months":
        return 12
    elif data.time_unit == "years":
        return 11
    else:
        raise ValueError("Invalid time unit")


def __get_seasonal_differencing_term(data: Dataset) -> int:
    """Get the seasonal differencing term for the SARIMA model"""
    maximum_differencing_term = 2
    logging.info(
        f"differencing term D for {data.name} is {ndiffs(data.values[data.subset_column_name], test='adf', max_d=maximum_differencing_term)}"
    )
    return ndiffs(
        data.values[data.subset_column_name],
        test="adf",
        max_d=maximum_differencing_term,
    )


def __get_trend_given_data(data: Dataset) -> str:
    """Get the trend given the data"""
    if data.name == "Sun spots":
        return "c"
    else:
        return "t"


def __get_number_of_lags_or_trend_autoregression_order(data: Dataset) -> int:
    """Get the number of lags or trend autoregression order for the SARIMA model"""
    if data.name == "Sun spots":
        return 3
    else:
        return 1


def __sarima_forecast(data: Dataset, config: dict) -> tuple:
    """Get the SARIMA forecast on the validation set"""
    training_set, validation_set, _ = __get_training_validation_and_test_sets(
        data
    )

    order, seasonal_order, trend = config

    # define model
    model = SARIMAX(
        training_set,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    # fit model
    model_fit = model.fit(disp=False)

    # make prediction
    prediction = model_fit.predict(
        start=len(training_set),
        end=len(training_set) + len(validation_set) - 1,
    )

    return prediction, validation_set


def __measure_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """Measure the RMSE of the SARIMA model"""
    # check if input contains NaN values
    if actual.isnull().values.any() or predicted.isnull().values.any():
        return np.nan
    return np.sqrt(mean_squared_error(actual, predicted))


def __validation(data: Dataset, config: dict) -> float:
    """Get the RMSE of the SARIMA model"""
    prediction, test_set = __sarima_forecast(data, config)
    return __measure_rmse(test_set, prediction)


def __score_model(
    data: Dataset, config: list, debug: bool = False
) -> Tuple[str, float]:
    """Score the SARIMA model"""

    result = None

    # convert config to a key
    key = str(config)
    # show all warnings and fail on exception if debugging
    if debug:
        result = __validation(data, config)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = __validation(data, config)
        except Exception as error:
            logging.error(f"Error: {error}")
            result = None
    # check for an interesting result
    if result is not None:
        logging.info(f" > Model[{key}] {result :.3f} RMSE")
        # store best result so far compared to global best
        # ensure not referenced before assignment
        if "best_score" not in globals():
            global best_score
            best_score = float("inf")
        if result < best_score:
            best_score = result
            logging.info("New best score: %.3f" % (best_score))

    return (config, result)


def __grid_search_configs(
    data: Dataset, cfg_list: list, parallel: bool = False
) -> list:
    """Grid search the SARIMA model"""
    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend="multiprocessing")
        tasks = (delayed(__score_model)(data, cfg, True) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [__score_model(data, cfg, True) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


def __get_sarima_configs(
    p_params: list = [0, 1, 2], # AR order
    d_params: list = [0, 1], # differencing order
    q_params: list = [0, 1, 2], # MA order
    t_params: list = ["n", "c", "t", "ct"], # trend
    large_p_params: list = [0, 1, 2], # seasonal AR order
    large_d_params: list = [0, 1], # seasonal differencing order
    large_q_params: list = [0, 1, 2], # seasonal MA order
    m_params: list = [0], # seasonal period
) -> List:
    """Get the SARIMA configurations"""
    model_configs = list()

    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in large_p_params:
                        for D in large_d_params:
                            for Q in large_q_params:
                                for m in m_params:
                                    # create and store config
                                    cfg = [(p, d, q), (P, D, Q, m), t]
                                    model_configs.append(cfg)
    return model_configs


def __get_best_sarima_model(data: Dataset) -> SARIMAX:
    """Get the best SARIMA model"""
    logging.info("Finding the best SARIMA model")
    if __stationarity(data):
        logging.info(f"Data {data.name} is stationary; no differencing required")
        d = 0
    else:
        logging.info(f"Data {data.name} is not stationary")
        d = __get_differencing_term(data)
        logging.info(f"Using differencing term {d}")
    
    seasonal = __get_seasonal_period(data)
    configs = __get_sarima_configs(m_params=[seasonal], d_params=[d])
    # grid search
    scores = __grid_search_configs(data, configs)
    logging.debug("done")
    # list top 3 configs
    for cfg, error in scores[:3]:
        logging.info(f"Config: {cfg}, RMSE: {error}")
    # get the parameters of the best model
    order, seasonal_order, trend = scores[0][0]

    # display best model parameters
    logging.info(
        f"Best SARIMA model parameters for {data.name}:\n=============================\norder: {order}\nseasonal_order: {seasonal_order}\ntrend: {trend}"
    )

    training_set = __get_training_set(data)

    # define model
    model = SARIMAX(
        training_set,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    # fit model
    model_fit = model.fit(disp=False)

    return model_fit


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

    p = __get_number_of_lags_or_trend_autoregression_order(data)

    my_order = (p, d, 0)
    my_seasonal_order = (1, D, 1, s)

    model = SARIMAX(
        __get_training_set(data),
        order=my_order,
        seasonal_order=my_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        trend=__get_trend_given_data(data),
    )
    logging.info(f"Number of observations for {data.name} is {len(data.values)}")
    logging.info(
        f"Training SARIMA model on {data.name} with order {my_order} and seasonal order {my_seasonal_order}"
    )
    if data.time_unit == "days":
        # show progress bar for model fitting
        return model.fit(disp=True)
    else:
        return model.fit(disp=False)


def __get_model_order_snake_case(model: Model) -> str:
    """Get the SARIMAXResults model order in snake case"""
    dict_of_model_orders = model.model_orders
    logging.info(
        f"Model orders for {model.model.data.orig_endog.name} are {dict_of_model_orders}"
    )
    return f"ar_{dict_of_model_orders['ar']}_trend_{dict_of_model_orders['trend']}_ma_{dict_of_model_orders['ma']}_seasonal_ar_{dict_of_model_orders['seasonal_ar']}_seasonal_ma_{dict_of_model_orders['seasonal_ma']}_exog_{dict_of_model_orders['exog']}"


def __forecast(model: Model, data: Dataset) -> pd.DataFrame:
    """Forecast the next 20% of the data"""
    title = f"{data.subset_column_name} forecast for {data.subset_row_name} for the next {__number_of_steps(data)} {data.time_unit} with SARIMA"
    logging.info(f"Forecasting {title}")
    return PredictionData(
        values=model.get_forecast(steps=__number_of_steps(data)).predicted_mean,
        prediction_column_name=None,
        ground_truth_values=__get_test_set(data),
        confidence_columns=None,
        title=title,
        plot_folder=f"{data.name}/{data.subset_row_name}/SARIMA/",
        plot_file_name=f"{data.subset_column_name}_forecast_{__get_model_order_snake_case(model)}",
    )


sarima = method(__get_best_sarima_model, __forecast)
