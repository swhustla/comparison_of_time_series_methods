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

We will use the pmdarima library to find the best SARIMA model for the data.
The pmdarima library is a Python implementation of the R library
"forecast" which is a popular library for time series analysis.
It includes a function called "auto_arima" which automatically finds the best
ARIMA model for the data. It also will perform a grid search to find the best
SARIMA model for the data.

"""

from typing import TypeVar, Generic, List, Tuple, Callable, Dict, Any
from methods.auto_arima import auto_arima as method
import pandas as pd


from multiprocessing import cpu_count

import logging

from arch.unitroot import ADF

import pmdarima as pm

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


def __stationarity(data: Dataset) -> bool:
    """Determines if the data is stationary"""
    data_to_check = data.values[data.subset_column_name]
    return ADF(data_to_check).pvalue < 0.05


def __get_seasonal_period(data: Dataset) -> int:
    """Get the seasonal period for the SARIMA model"""

    if not data.seasonality:
        return 0

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



def __fit_pmdarima_model(
    data: Dataset
) -> pd.Series:
    """Evaluate the pmdarima forecast on the training set
    See here: https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html?highlight=pmdarima
    """
    training_data = __get_training_set(data)
    logging.info(f"Defining pmdarima model with seasonal={data.seasonality} and seasonal_period={__get_seasonal_period(data)}, maxiter=100")
    # define model
    # if seasonal, force D=1
    capital_d = 1 if data.seasonality else 0

    # show frequency of data
    logging.info(f"Frequency of training_data: {training_data.index.inferred_freq}")

    model = pm.auto_arima(
        y=training_data,
        stationary=__stationarity(data),
        start_p=0,
        d=1,
        start_q=0,
        test="adf",
        max_p=5,
        max_d=5,
        max_q=5,
        m=__get_seasonal_period(data), # seasonal period, or frequency of the data
        seasonal=data.seasonality,
        start_P=0,
        D=capital_d,
        start_Q=0,
        max_P=5,
        max_D=5,
        max_Q=5,
        trace=True,
        error_action="ignore",
        suppress_warnings=True,
        # information_criterion="aic",
        stepwise=False,
        n_jobs=cpu_count(),
        n_fits=50,
        # maxiter=100,
    )

    print(model.summary())

    # fit model
    logging.info("Fitting pmdarima model:")

    return model.fit(y=training_data)



def __get_pmdarima_model_order_snake_case(model: Model) -> str:
    """Get the pmdarima model order in snake case"""
    dict_of_model_orders = model.get_params()["order"]
    logging.info(
        f"Model orders for SARIMA (pmdarima) are {dict_of_model_orders}"
    )
    logging.info(f"AR params are {model.arparams()}")
    logging.info(f"MA params are {model.maparams()}")

    # TODO: get P, Q, D, s parameters from pmdarima model

    return f"ar_{dict_of_model_orders[0]}_trend_{dict_of_model_orders[1]}_ma_{dict_of_model_orders[2]}"


def __forecast_pmdarima(model: Model, data: Dataset) -> pd.DataFrame:
    """Forecast the next 20% of the data"""
    title = f"{data.subset_column_name} forecast for {data.subset_row_name} for the next {__number_of_steps(data)} {data.time_unit} with Auto ARIMA"
    prediction_in_sample = model.predict_in_sample()
    print(f"Length of in-sample prediction: {len(prediction_in_sample)}")
    print(f"Number of steps: {__number_of_steps(data)}")
    logging.info(f"Forecasting {title}")
    return PredictionData(
        method_name="auto_arima",
        values=model.predict(__number_of_steps(data)),
        prediction_column_name=None,
        ground_truth_values=__get_test_set(data),
        confidence_columns=None,
        title=title,
        plot_folder=f"{data.name}/{data.subset_row_name}/auto_arima/",
        plot_file_name=f"{data.subset_column_name}_forecast_{__get_pmdarima_model_order_snake_case(model)}",
        number_of_iterations=model.get_params()["maxiter"],
        color="darkorange",
        in_sample_prediction=prediction_in_sample,
    )


auto_arima = method(__fit_pmdarima_model, __forecast_pmdarima)
