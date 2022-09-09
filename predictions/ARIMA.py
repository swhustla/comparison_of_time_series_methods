from typing import TypeVar
from methods.ARIMA import arima as method
import pandas as pd

from arch.unitroot import KPSS, ADF
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.forecasting.stl import STLForecast
import statsmodels.api as sm

from data.Data import Dataset, Result
from predictions.Prediction import PredictionData

Model = TypeVar("Model")


def __number_of_steps(data: Dataset) -> int:
    return len(data.values) // 5


def __get_training_set(data: Dataset) -> pd.DataFrame:
    return data.values[: -__number_of_steps(data)]


def __get_test_set(data: Dataset) -> pd.DataFrame:
    return data.values[-__number_of_steps(data) :]


def __stationarity(data: Dataset) -> bool:
    """Determines if the data is stationary"""
    return ADF(data.values).pvalue < 0.05


def __get_differencing_term(data: Dataset) -> int:
    """Get the differencing term for the ARIMA model"""
    return ndiffs(data.values, test="adf")


def __fit_auto_regressive_model(data: Dataset) -> Model:
    """Fit an ARIMA model to the first 80% of the data"""
    model = STLForecast(
        __get_training_set(data),
        sm.tsa.arima.ARIMA,
        model_kwargs=dict(order=(1, __get_differencing_term(data), 0), trend="t"),
    )

    model_result = model.fit().model_result
    return model_result


def __forecast(model: Model, data:Dataset) -> PredictionData:
    """ Forecast the next 20% of the data """
    title = f"{data.subset_column_name} forecast for {data.subset_row_name} with ARIMA"
    return PredictionData(
        values=model.get_forecast(steps=__number_of_steps(data)).summary_frame(),
        prediction_column_name=None,
        ground_truth_values=__get_test_set(data),
        confidence_columns=["mean_ci_lower", "mean_ci_upper"],
        title=title,
    )


arima = method(
    __stationarity, __fit_auto_regressive_model, __forecast
)
