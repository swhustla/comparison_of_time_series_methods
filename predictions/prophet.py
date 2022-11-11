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


from typing import TypeVar, List, Dict, Generator
from prophet import Prophet
import pandas as pd
import numpy as np

import logging

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


def __get_configs() -> Generator:
    """Get the configs for a grid search on Prophet"""
    for changepoint_range in [0.5, 0.8, 0.95]:
        for seasonality_prior_scale in [0.01, 0.1, 1.0]:
            for changepoint_prior_scale in [0.01, 0.1, 1.0]:
                for seasonality_mode in ["additive", "multiplicative"]:
                    yield {
                        "changepoint_range": changepoint_range,
                        "seasonality_prior_scale": seasonality_prior_scale,
                        "changepoint_prior_scale": changepoint_prior_scale,
                        "seasonality_mode": seasonality_mode,
                    }



def __measure_mape(actual: pd.DataFrame, predicted: np.array) -> float:
    """Measure the Mean Absolute Percentage Error"""
    # change actual to a numpy array
    actual = actual.drop(columns=["Date"]).values
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def __score_model(model: Model, data: Dataset) -> float:
    """Score the model on the last 20% of the data"""
    test_df = __get_test_set(data)
    future_dates = __get_future_dates(data)
    forecast = model.predict(future_dates)
    return __measure_mape(test_df, forecast["yhat"].values)


def __get_best_model(data: Dataset) -> Model:
    """Get the best model based on the configs"""
    best_score = float("inf")
    best_model = None
    for config in __get_configs():
        logging.info(f"Trying config: {str(config)}")
        try:
            model = __fit_prophet_model(data, config)
        except Exception as e:
            logging.error(f"Error: {e}")
            continue
        score = __score_model(model, data)
        if score < best_score:
            logging.info(f"New best score: {score}")
            best_score = score
            best_model = model
        logging.info(f"Best config: {config} -> {best_score}")
    return best_model



# TODO: Add settings for the model to include seasonality, holidays, etc.
def __fit_prophet_model(data: Dataset, config:dict) -> Model:
    """Fit a Prophet model to the first 80% of the data"""
    train_df = __get_training_set(data)
    daily_seasonality = __check_if_data_is_seasonal_this_time_unit(data, "hours")
    weekly_seasonality = __check_if_data_is_seasonal_this_time_unit(data, "days")
    yearly_seasonality = __check_if_data_is_seasonal_this_time_unit(data, "months")

    model = Prophet(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        seasonality_prior_scale=config["seasonality_prior_scale"],
        changepoint_range=config["changepoint_range"],
        changepoint_prior_scale=config["changepoint_prior_scale"],
        seasonality_mode=config["seasonality_mode"],
    )

    # in the case of monthly data, Prophet will automatically detect seasonality


    renamed_df = train_df.rename(columns={"Date": "ds", data.subset_column_name: "y"})
    model.fit(renamed_df)
    return model


def __get_future_dates(data: Dataset) -> pd.DataFrame:
    """# construct a dataframe with the future dates"""
    # TODO: Ensure that the frequency is correct (e.g. daily, weekly, monthly, etc.)
    future_dates = __get_test_set(data)["Date"]
    datetime_version = pd.to_datetime(future_dates)
    return pd.DataFrame({"ds": datetime_version})


def __check_if_data_is_seasonal_this_time_unit(data: Dataset, time_unit: str) -> bool:
    """Check if the data is daily"""
    if data.seasonality == True:
        return data.time_unit == time_unit
    else:
        return False


def __forecast(model: Model, data: Dataset) -> PredictionData:
    """Forecast the next 20% of the data"""
    title = (
        f"{data.subset_column_name} forecast for {data.subset_row_name} with Prophet"
    )
    future = __get_future_dates(data)
    # TODO: Add settings for the model to include seasonality, holidays, etc.
    # ideally changepoint_range=1.0, changepoint_prior_scale=0.05 (Suman used 0.75)
    # daily_seasonality=True

    forecast = model.predict(future)
    forecast_df = forecast.set_index(keys=["ds"])
    ground_truth_df = (
        __get_test_set(data)
        .rename(columns={"Date": "ds", data.subset_column_name: "y"})
        .set_index(keys=["ds"])
    )
    return PredictionData(
        values=forecast_df,
        prediction_column_name="yhat",
        ground_truth_values=ground_truth_df["y"],
        confidence_columns=["yhat_lower", "yhat_upper"],
        title=title,
        plot_folder=f"{data.name}/{data.subset_row_name}/Prophet/",
        plot_file_name=f"{data.subset_column_name}_forecast",
    )


prophet = method(__get_best_model, __forecast)
