"""
The Auto Regressive Prediction method

Auto-regressive models are a class of statistical models for analyzing and forecasting time series data. 
They explicitly model the relationship between the observations and their lagged values. 
This is in contrast to other methods that model the relationship between the observations and a deterministic trend, such as linear or exponential trend.
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


from typing import TypeVar, Callable, Tuple
from methods.AR import ar as method
import pandas as pd

import logging

from plots.color_map_by_method import get_color_map_by_method

from arch.unitroot import ADF
from statsmodels.tsa.ar_model import ar_select_order
from statsmodels.tsa.forecasting.stl import STLForecast
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error

from data.dataset import Dataset, Result
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
    Returns the maximum number of lags to use for determining the lag order.
    """
    if data.time_unit == "years":
        return 13
    elif data.time_unit == "months":
        return 24
    elif data.time_unit == "weeks":
        return 52
    elif data.time_unit == "days":
        return 365


def __transform_data(data: Dataset) -> pd.DataFrame:
    """
    Transforms the data to a pandas DataFrame.
    """
    return pd.DataFrame(data.values[data.subset_column_name])


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


def __determine_lag_order(data: Dataset) -> int:
    """
    Determines the lag order of the model.
    https://www.statsmodels.org/stable/generated/statsmodels.tsa.ar_model.ar_select_order.html
    """
    max_lag = __get_number_of_lags(data)
    logging.info(f"Max lag: {max_lag}")
    if data.seasonality:
        max_lag = 1
        ar_order = ar_select_order(
            __get_training_set(data),
            maxlag=max_lag,
            seasonal=data.seasonality,
            period=__get_period_of_seasonality(data),
        )
    else:
        ar_order = ar_select_order(__get_training_set(data), maxlag=max_lag)
    logging.info(
        f"AR lag order for {data.name} {data.subset_column_name}: {ar_order.ar_lags}"
    )
    return ar_order.ar_lags[0]


def __evaluate_auto_regressive_model(
    data: Dataset, arima_order: tuple, trend: str
) -> Result:
    """Evaluate an ARIMA model for a given order (p,d,q) and return RMSE"""

    # set up the model and fit it
    model = STLForecast(
        endog=__get_training_set(data),
        period=__get_period_of_seasonality(data),
        model=sm.tsa.ARIMA,
        model_kwargs={"order": arima_order, "trend": trend},
    )
    model_fit = model.fit()

    # return in sample bayesian information criterion
    return model_fit.model_result.bic


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
                        evaluation_metric = __evaluate_auto_regressive_model(
                            data, arima_order=order, trend=t
                        )
                        if evaluation_metric < best_score:
                            best_score, best_cfg, best_trend = evaluation_metric, order, t
                        logging.info(f"ARIMA{order} Trend={t} BIC={evaluation_metric}")
                    except Exception as e:
                        logging.error(f"ARIMA{order} Trend={t} failed with error: {e}")
                        continue
    logging.info(f"Best AR: {best_cfg} Best trend: {best_trend} BIC={best_score}")
    return best_cfg, best_trend


__p_values = [0, 1, 2, 4, 8, 10] # auto-regressive order
__d_values = [0] # integrated order
__q_values = [0] # moving average order
__trend_values = ["c", "t", "ct"] # trend

def __calculate_number_of_configurations() -> int:
    """Calculate the number of configurations to be tested"""
    return len(__p_values) * len(__d_values) * len(__q_values) * len(__trend_values)


def __get_best_model_order(data: Dataset) -> Model:
    """Get the best model order for the ARIMA model"""
    return __evaluate_models(data, __p_values, __d_values, __q_values, __trend_values)


def __train_auto_regressive_model(data: Dataset) -> Model:
    """
    Fit an AR model, chosen by the best model order.
    """
    (ar_order, int_order, ma_order), trend = __get_best_model_order(data)


    model = STLForecast(
        endog=__get_training_set(data),
        model=sm.tsa.ARIMA,
        model_kwargs=dict(order=(ar_order, int_order, ma_order), trend=trend),
        period=__get_period_of_seasonality(data),
    )
    return model.fit()


def __get_model_order_snake_case(model: Model) -> str:
    """convert model order dict to snake case filename"""

    model_order = model.model.order
    model_order = f"AR{model_order[0]}_I{model_order[1]}_MA{model_order[2]}"
    return model_order.replace(" ", "_")


def __forecast(model: Model, data: Dataset) -> pd.DataFrame:
    """
    Makes a forecast using the trained model.
    """
    title = f"{data.subset_column_name} for {data.subset_row_name} forecast using Auto Regressive model"

    prediction = model.forecast(__number_of_steps(data))
    length_in_sample = len(__get_training_set(data).values)
    prediction_in_sample = model.get_prediction(0,length_in_sample).summary_frame()
    prediction_summary = model.model_result.get_forecast(
        __number_of_steps(data)
    ).summary_frame()
    combined_data = pd.concat([prediction, prediction_summary], axis=1)
    combined_data.rename(columns={0: "forecast"}, inplace=True)

    return PredictionData(
        method_name="AR",
        values=combined_data,
        prediction_column_name="forecast",
        ground_truth_values=__get_test_set(data),
        confidence_columns=["mean_ci_lower", "mean_ci_upper"],
        title=title,
        plot_folder=f"{data.name}/{data.subset_row_name}/AR/",
        plot_file_name=f"{data.subset_column_name}_forecast_{__get_model_order_snake_case(model)}",
        number_of_iterations=__calculate_number_of_configurations(),
        confidence_on_mean=True,
        confidence_method="95% confidence interval",
        color=get_color_map_by_method("AR"),
        in_sample_prediction=prediction_in_sample.iloc[:, 0],
    )

# TODO: add grid search for AR model
ar = method(
    __quick_check_for_auto_correlation,
    __train_auto_regressive_model,
    __forecast,
)
