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
from sklearn.metrics import mean_absolute_percentage_error
import pandas as pd
import numpy as np

import logging
# set up logging
logging.basicConfig(level=logging.DEBUG)

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


#TODO: Add support for multiple time series

def __get_configs() -> Generator:
    """Get the configs for a grid search on Prophet"""
    for changepoint_range in [0.1, 0.9]:#[0.01, 0.1, 0.9, 0.99]:
        for seasonality_prior_scale in [0.1, 0.9]: #[0.1, 0.9, 0.99, 1.0]:
            for changepoint_prior_scale in [0.001, 0.05, 0.5, 0.95, 0.99]:
                for seasonality_mode in ["additive", "multiplicative"]:
                    yield {
                        "changepoint_range": changepoint_range,
                        "seasonality_prior_scale": seasonality_prior_scale,
                        "changepoint_prior_scale": changepoint_prior_scale,
                        "seasonality_mode": seasonality_mode,
                    }



def __measure_error_metric(data: Dataset, actual: pd.DataFrame, predicted: np.array) -> float:
    """Measure the error"""
    # change actual to a numpy array
    actual_this = actual.loc[:, data.subset_column_name].values
    print(f"Actual: {actual_this[:10]}")
    print(f"Predicted: {predicted[:10]}")
    result = mean_absolute_percentage_error(y_true=actual_this, y_pred=predicted)
    print(f"Error: {result}")
    return result
    

def __score_model(model: Model, data: Dataset) -> float:
    """Score the model on a test set"""
    forecast = model.predict()
    return __measure_error_metric(data, __get_training_set(data), forecast["yhat"].values)


def __get_best_model(data: Dataset) -> Tuple[Model, int]:
    """Get the best model based on the configs"""
    best_score = float("inf")
    best_model = None
    configs = __get_configs()
    
    training_df = __get_training_set(data)
    renamed_training_df = training_df.rename(columns={"Date": "ds", data.subset_column_name: "y"})
    """Removes the time from datetime"""
    renamed_training_df_notimezone = pd.to_datetime(renamed_training_df['ds']).dt.tz_localize(None)
    renamed_training_df = pd.concat([renamed_training_df_notimezone, renamed_training_df['y']], axis=1)

    for config in configs:
        model = None
        print(f"Trying config: \n{str(config)} \nfor dataset {data.name} - {data.subset_row_name}")
        try:
            model = __fit_prophet_model(renamed_training_df, config)
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e
            
            
        score = __score_model(model, data)

        # print score if not nan
        if not np.isnan(score):
            logging.info(f"\n\nScore: {score}")

        if score < best_score:
            print(f"New best score: {score}")
            best_score = score
            best_model = model
            best_config = config
        
        del model
    # log the best config formatted nicely
    formatted_config = ", ".join([f"{key}={value}" for key, value in best_config.items()])
    print(f"\n\nBest config: {formatted_config} -> {best_score}")
    # get length of the configs geenrator
    return best_model, len(list(configs))


def __get_seasonality_settings(data: Dataset) -> Dict:
    """Get the seasonality settings for the model"""
    settings = {}
    for time_unit in ["hours", "days", "weeks", "months", "years"]:
        if __check_if_data_is_seasonal_this_time_unit(data, time_unit):
            settings[time_unit] = True
        else:
            settings[time_unit] = False

    if not settings["years"] and settings["months"]:
        settings["years"] = True

    return settings


# TODO: Add settings for the model to include seasonality, holidays, etc.
def __fit_prophet_model(training_data: pd.DataFrame, config:dict) -> Model:
    """Fit a Prophet model to the first 80% of the data"""

    model = Prophet(
        seasonality_prior_scale=config["seasonality_prior_scale"],
        changepoint_range=config["changepoint_range"],
        changepoint_prior_scale=config["changepoint_prior_scale"],
        seasonality_mode=config["seasonality_mode"],
    )

    model.fit(df=training_data)
    return model


def __get_future_dates(data: Dataset) -> pd.DataFrame:
    """# construct a dataframe with the future dates"""
    # TODO: Ensure that the frequency is correct (e.g. daily, weekly, monthly, etc.)

    test_set = __get_test_set(data)
    future_dates = test_set["Date"]
    datetime_version = pd.to_datetime(future_dates)
    return pd.DataFrame({"ds": datetime_version})

def __get_training_dates(data: Dataset) -> pd.DataFrame:
    """# construct a dataframe with the future dates"""
    # TODO: Ensure that the frequency is correct (e.g. daily, weekly, monthly, etc.)

    training_set = __get_training_set(data)
    training_dates = training_set["Date"]
    datetime_version = pd.to_datetime(training_dates)
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
    future = __get_future_dates(data)
    """Removes the time from datetime"""
    future = future['ds'].dt.tz_localize(None)
    in_sample = __get_training_dates(data)
    """Removes the time from datetime"""
    in_sample = in_sample['ds'].dt.tz_localize(None)
    future_in_sample = pd.concat([in_sample,future])
    future_in_sample = pd.to_datetime(future_in_sample)
    future_in_sample = pd.DataFrame({"ds": future_in_sample})
    

    # TODO: Add settings for the model to include holidays, etc.

    forecast_in_sample_plus_future = model.predict(df=future_in_sample)
    print(f" forecast_in_sample_plus_future\n{ forecast_in_sample_plus_future}")
    forecast = forecast_in_sample_plus_future[-__number_of_steps(data):]
    predict_in_sample = forecast_in_sample_plus_future[:len(__get_training_set(data).values)+2]
    forecast_df = forecast.set_index(keys=["ds"])
    predict_in_sample_df = predict_in_sample.set_index(keys=["ds"])
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
        in_sample_prediction=predict_in_sample_df["yhat"],
    )


prophet = method(__get_best_model, __forecast)
