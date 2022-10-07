""" 
Facebook Prophet 

    https://facebook.github.io/prophet/docs/quick_start.html#python-api

This is a wrapper around the Prophet library, released by Facebook in 2017.
A quick summary of the library is as follows: 
    Prophet is a procedure for forecasting time series data based on an additive model 
    where non-linear trends are fit with yearly, weekly, and daily seasonality, plus 
    holiday effects.
    It has been used for forecasting retail foot traffic with RMSPE of 25% on a 
    hold-out set, and for macroeconomic forecasting with MAPE of 2.5% on a hold-out set.
    It works best with daily periodicity data with at least one year of historical data.
    Prophet is robust to missing data and shifts in the trend, and typically handles outliers
    well.

    A great introduction to Prophet can be found on Towards Data Science:
    https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b


Downsides of Prophet:
    - It is not very flexible, and it is not possible to change the model parameters
    - It is not possible to use it for forecasting multiple time series at the same time

"""


from typing import TypeVar
from prophet import Prophet
import pandas as pd

from methods.prophet import prophet as method

from data.dataset import Dataset
from predictions.Prediction import PredictionData

Model = TypeVar("Model")

def __get_dataframe_with_date_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe.reset_index().rename(columns={"date": "Date"})

def __number_of_steps(data: Dataset) -> int:
    return int(len(data.values) // 5)


def __get_training_set(data: Dataset) -> pd.DataFrame:
    return __get_dataframe_with_date_column(data.values[: -__number_of_steps(data)])


def __get_test_set(data: Dataset) -> pd.DataFrame:
    return __get_dataframe_with_date_column(data.values[-__number_of_steps(data) :])

#TODO: Add settings for the model to include seasonality, holidays, etc.
def __fit_prophet_model(data: Dataset) -> Model:
    """Fit a Prophet model to the first 80% of the data"""
    train_df = __get_training_set(data)
    model = Prophet()
    renamed_df = train_df.rename(columns={"Date": "ds", data.subset_column_name: "y"})
    model.fit(renamed_df)
    return model

def __get_future_dates(data: Dataset) -> pd.DataFrame:
    """# construct a dataframe with the future dates"""
    future_dates = __get_test_set(data)["Date"]
    datetime_version = pd.to_datetime(future_dates)
    return pd.DataFrame({"ds": datetime_version})

def __forecast(model: Model, data:Dataset) -> PredictionData:
    """ Forecast the next 20% of the data """
    title = f"{data.subset_column_name} forecast for {data.subset_row_name} with Prophet"
    future = __get_future_dates(data)
    forecast = model.predict(future)
    forecast_df = forecast.set_index(keys=["ds"])
    ground_truth_df = __get_test_set(data).rename(columns={"Date": "ds", data.subset_column_name: "y"}).set_index(keys=["ds"])
    return PredictionData(
        values=forecast_df,
        prediction_column_name="yhat",
        ground_truth_values=ground_truth_df["y"],
        confidence_columns=["yhat_lower", "yhat_upper"],
        title=title,
        plot_folder=f"{data.name}/{data.subset_row_name}/Prophet/",
        plot_file_name=f"{data.subset_column_name}_forecast",
    )

prophet = method(__fit_prophet_model, __forecast)
