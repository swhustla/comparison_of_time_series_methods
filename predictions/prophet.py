from os import rename
from time import time
from typing import TypeVar
from prophet import Prophet
import pandas as pd

from methods.prophet import prophet as method

from data.Data import Dataset, Result
from predictions.Prediction import PredictionData

Model = TypeVar("Model")

def __get_dataframe_with_date_column(dataframe: pd.Dataframe) -> pd.DataFrame:
    return dataframe.reset_index()

def __number_of_steps(data: Dataset) -> int:
    return len(data.values) // 5


def __get_training_set(data: Dataset) -> pd.DataFrame:
    return __get_dataframe_with_date_column(data.values[: -__number_of_steps(data)])


def __get_test_set(data: Dataset) -> pd.DataFrame:
    return __get_dataframe_with_date_column(data.values[-__number_of_steps(data) :])

def __fit_prophet_model(data: Dataset) -> Model:
    """Fit a Prophet model to the first 80% of the data"""
    df = __get_training_set(data)
    model = Prophet()
    model.fit(df.rename(columns={"index": "ds", data.subset_column_name: "y"}))
    return model

def __get_future_dates(data: Dataset) -> pd.DataFrame:
    """# construct a dataframe with the future dates"""
    future_dates = __get_test_set(data).index
    return pd.DataFrame({"ds": future_dates})

def __forecast(model: Model, data:Dataset) -> PredictionData:
    """ Forecast the next 20% of the data """
    title = f"{data.subset_column_name} forecast for {data.subset_row_name} with Prophet"
    future = __get_future_dates(data)
    forecast = model.predict(future)
    return PredictionData(
        values=forecast,
        ground_truth_values=__get_test_set(data),
        confidence_columns=["yhat_lower", "yhat_upper"],
        title=title,
    )

prophet = method(__fit_prophet_model, __forecast)
