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

    Prophet estimates uncertainty intervals as well as forecasts. For uncertainty intervals, it uses Monte Carlo sampling.
    The uncertainty intervals are not guaranteed to be accurate, but they should be a reasonable fit for most applications.
    The default uncertainty interval is 80%.

    A great introduction to Prophet can be found on Towards Data Science:
    https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b


Downsides of Prophet:
    - It is not very flexible, and it is not possible to change the model parameters
    - It is not possible to use it for forecasting multiple time series at the same time

"""


from typing import TypeVar, List, Dict, Generator, Tuple
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


def __get_training_and_test_set(data: Dataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get the training and test set"""
    training_set = __get_training_set(data)
    test_set = __get_test_set(data)
    return training_set, test_set

def __get_training_test_and_validation_set(data: Dataset) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get the training, test and validation set"""
    training_set, test_set = __get_training_and_test_set(data)
    validation_set = training_set[-__number_of_steps(data) :]
    training_set = training_set[: -__number_of_steps(data)]
    return training_set, validation_set, test_set

#TODO: Add support for multiple time series

def __get_configs() -> Generator:
    """Get the configs for a grid search on Prophet"""
    for changepoint_range in [0.01, 0.1, 1.0]:
        for seasonality_prior_scale in [0.1, 1.0, 10]:
            for changepoint_prior_scale in [0.01, 0.1, 1.0]:
                for seasonality_mode in ["additive", "multiplicative"]:
                    yield {
                        "changepoint_range": changepoint_range,
                        "seasonality_prior_scale": seasonality_prior_scale,
                        "changepoint_prior_scale": changepoint_prior_scale,
                        "seasonality_mode": seasonality_mode,
                    }



def __measure_mape(data: Dataset, actual: pd.DataFrame, predicted: np.array) -> float:
    """Measure the Mean Absolute Percentage Error"""
    # change actual to a numpy array
    actual = actual.loc[:, data.subset_column_name].values
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def __score_model(model: Model, data: Dataset, use_validation: bool=False) -> float:
    """Score the model on a validation set or test set"""
    if use_validation:
        _, validation_set, _ = __get_training_test_and_validation_set(data)
    else:
        validation_set = __get_test_set(data)
    future_dates = __get_future_dates(data, use_validation=use_validation)
    forecast = model.predict(future_dates)
    return __measure_mape(data, validation_set, forecast["yhat"].values)


def __get_best_model(data: Dataset) -> Tuple[Model, int]:
    """Get the best model based on the configs"""
    best_score = float("inf")
    best_model = None
    configs = __get_configs()
    for config in configs:
        logging.info(f"Trying config: {str(config)} for dataset {data.name} - {data.subset_row_name}")
        try:
            model = __fit_prophet_model(data, config)
        except Exception as e:
            logging.error(f"Error: {e}")
            continue
        score = __score_model(model, data, use_validation=True)
        if score < best_score:
            logging.info(f"New best score: {score}")
            best_score = score
            best_model = model
    logging.info(f"Best config: {config} -> {best_score}")
    # get length of the configs geenrator
    return best_model, len(list(configs))



# TODO: Add settings for the model to include seasonality, holidays, etc.
def __fit_prophet_model(data: Dataset, config:dict) -> Model:
    """Fit a Prophet model to the first 80% of the data"""
    train_df = __get_training_set(data)
    daily_seasonality = __check_if_data_is_seasonal_this_time_unit(data, "hours")
    weekly_seasonality = __check_if_data_is_seasonal_this_time_unit(data, "days")
    yearly_seasonality = __check_if_data_is_seasonal_this_time_unit(data, "months")

    monthly_seasonality = __check_if_data_is_seasonal_this_time_unit(data, "weeks")

    if not yearly_seasonality and  monthly_seasonality:
        logging.info("Monthly seasonality detected")
        yearly_seasonality = True

    model = Prophet(
        daily_seasonality=daily_seasonality,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        seasonality_prior_scale=config["seasonality_prior_scale"],
        changepoint_range=config["changepoint_range"],
        changepoint_prior_scale=config["changepoint_prior_scale"],
        seasonality_mode=config["seasonality_mode"],
    )

    # in the case of weekly data, with monthly and yearly seasonalty then we add the seasonality manually
    if monthly_seasonality and yearly_seasonality:
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    renamed_df = train_df.rename(columns={"Date": "ds", data.subset_column_name: "y"})
    model.fit(renamed_df)
    return model


def __get_future_dates(data: Dataset, use_validation: bool = False) -> pd.DataFrame:
    """# construct a dataframe with the future dates"""
    # TODO: Ensure that the frequency is correct (e.g. daily, weekly, monthly, etc.)
    if use_validation:
        _, validation_set, _ = __get_training_test_and_validation_set(data)
        future_dates = validation_set["Date"]
    else:
        test_set = __get_test_set(data)
        future_dates = test_set["Date"]
    datetime_version = pd.to_datetime(future_dates)
    return pd.DataFrame({"ds": datetime_version})


def __check_if_data_is_seasonal_this_time_unit(data: Dataset, time_unit: str) -> bool:
    """Check if the data is or other time unit is seasonal"""
    if data.seasonality == True:
        return data.time_unit == time_unit
    else:
        return False


def __forecast(model: Model, data: Dataset, number_of_configs: int) -> PredictionData:
    """Forecast the next 20% of the data"""
    title = (
        f"{data.subset_column_name} forecast for {data.subset_row_name} with Prophet"
    )
    future = __get_future_dates(data, use_validation=False)
    # TODO: Add settings for the model to include holidays, etc.

    forecast = model.predict(future)
    forecast_df = forecast.set_index(keys=["ds"])
    ground_truth_df = (
        __get_test_set(data)
        .rename(columns={"Date": "ds", data.subset_column_name: "y"})
        .set_index(keys=["ds"])
    )
    return PredictionData(
        method_name="Prophet",
        values=forecast_df,
        prediction_column_name="yhat",
        ground_truth_values=ground_truth_df["y"],
        confidence_columns=["yhat_lower", "yhat_upper"],
        title=title,
        plot_folder=f"{data.name}/{data.subset_row_name}/Prophet/",
        plot_file_name=f"{data.subset_column_name}_forecast",
        number_of_iterations=number_of_configs,
        confidence_on_mean=False,
        confidence_method="80% confidence interval by Monte Carlo sampling",
        color="violet",
    )


prophet = method(__get_best_model, __forecast)
