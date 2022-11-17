"""Simple Moving Average Prediction


Moving average time series models are a type of linear regression model that
uses the average of the previous observations as the prediction for the next
time period. The moving average model is a type of ARIMA model, where the
autoregressive part of the model is set to zero. The moving average model is
useful for removing the trend and seasonality from the data, and is often used
as a preprocessing step for other models.


The simple MA method is a special case of the general linear regression model, where the regressors are lagged values of the dependent variable. The simple MA model is also a special case of the autoregressive moving average model, where the autoregressive coefficients are all zero.
It is defined by the following equation:

y_t = \frac{1}{n} \sum_{i=1}^n y_{t-i}

where y_t is the value of the time series at time t, and 
n is the number of values to average.

For example, if n = 3, then the simple MA model is defined by the following equation:

y_t = \frac{1}{3} (y_{t-1} + y_{t-2} + y_{t-3})

The simple MA model is a good baseline to compare other methods against.
It is equivalent to using an ARMA model with no autoregressive terms and 
a moving average term of order n.

The best Python library for simple MA models is statsmodels. It provides a wide range of models, including AR, ARMA, ARIMA, ARIMAX, VAR, VARMAX, and SARIMAX. It also provides a wide range of tools for model selection, diagnostics, and visualization.

"""


from typing import TypeVar, Callable
from methods.MA import ma as method
import pandas as pd

from arch.unitroot import ADF
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.forecasting.stl import STLForecast

import statsmodels.api as sm

from matplotlib import pyplot as plt

from data.dataset import Dataset, Result
from predictions.Prediction import PredictionData

Model = TypeVar("Model")


def __number_of_steps(data: Dataset) -> int:
    return int(len(data.values) // 5)


def __get_training_set(data: Dataset) -> pd.DataFrame:
    return data.values[: -__number_of_steps(data)][data.subset_column_name]


def __get_test_set(data: Dataset) -> pd.DataFrame:
    return data.values[-__number_of_steps(data) :][data.subset_column_name]


def __check_for_stationarity(data: Dataset) -> bool:
    """
    Checks if the data is stationary using the Augmented Dickey-Fuller test.
    A p-value of less than 0.05 indicates that the data is stationary.
    """
    adf = ADF(data.values[data.subset_column_name], method="AIC")
    return adf.pvalue < 0.05


def __transform_data(data: Dataset) -> pd.DataFrame:
    """
    Transforms the data to make it stationary.
    """
    if __check_for_stationarity(data):
        return data.values[data.subset_column_name]
    else:
        return data.values[data.subset_column_name].diff()


def __get_number_of_lags(data: Dataset) -> int:
    """
    Returns the number of lags to use in the auto-regressive model.
    The number of lags is important for the model to be able to
    capture the auto-correlation in the data.
    """
    if data.time_unit == "days":
        return 364
    elif data.time_unit == "weeks":
        return 51
    elif data.time_unit == "months":
        return 12
    elif data.time_unit == "years":
        return 11
    else:
        return 1


def __check_for_auto_correlation(data: Dataset) -> bool:
    """
    Checks if the data has auto-correlation using the plot_acf function.
    """
    plot_acf(data.values[data.subset_column_name])
    plt.show()
    return input("Does the data have auto-correlation? (y/n): ") == "y"


def __fit_simple_ma(data: Dataset, ma_order: int = 1) -> Model:
    """
    Fits the simple MA model to the first 80% of the data.
    """
    # We can use the ARIMA class to create an MA model and
    # setting a zeroth-order AR model and an integration order of 0.
    # We must specify the order of the MA model in the order argument.
    ar_order = 0
    integ_order = 0
    model = STLForecast(
        __get_training_set(data),
        model=sm.tsa.ARIMA,
        model_kwargs=dict(order=(ar_order, integ_order, ma_order), trend="t"),
        period=__get_number_of_lags(data),
    )

    return model.fit()


def __get_model_order_snake_case(model: Model) -> str:
    """convert model order dict to snake case filename"""

    model_order = model.model.order
    model_order = f"AR{model_order[0]}_I{model_order[1]}_MA{model_order[2]}"
    return model_order.replace(" ", "_")


def __forecast(model: Model, data: Dataset) -> PredictionData:
    """
    Makes a prediction for the next 20% of the data using the fitted model.
    """
    title = f"{data.subset_column_name} forecast for {data.subset_row_name} with simple MA"
    prediction = model.forecast(__number_of_steps(data))
    prediction_summary = model.model_result.get_forecast(__number_of_steps(data)).summary_frame()
    combined_data = pd.concat([prediction, prediction_summary], axis=1)
    combined_data.rename(columns={0: "forecast"}, inplace=True)

    return PredictionData(
        values=combined_data,
        prediction_column_name="forecast",
        ground_truth_values=__get_test_set(data),
        confidence_columns=["mean_ci_lower", "mean_ci_upper"],
        title=title,
        plot_folder=f"{data.name}/{data.subset_row_name}/MA/",
        plot_file_name=f"{data.subset_column_name}_forecast_{__get_model_order_snake_case(model)}",
    )

# TODO: add grid search for MA order
ma = method(__fit_simple_ma, __forecast)
