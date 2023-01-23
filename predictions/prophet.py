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


from typing import TypeVar, List, Dict, Generator, Tuple, Union, Optional
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

from plots.color_map_by_method import get_color_map_by_method

import holidays
import pycountry

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


# TODO: Add support for multiple time series


def __get_configs(seasonality_prior_scale_list: List[int] = [10]) -> Generator:
    """Get the configs for a grid search on Prophet"""

    if len(seasonality_prior_scale_list) == 0:
        for changepoint_range in [0.5, 0.8]:
            for changepoint_prior_scale in [0.01, 0.05]:
                yield {
                    "changepoint_range": changepoint_range,
                    "changepoint_prior_scale": changepoint_prior_scale,
                }

    for changepoint_range in [0.5, 0.8]:  # [0.01, 0.1, 0.9, 0.99]:
        for seasonality_prior_scale in seasonality_prior_scale_list:
            for changepoint_prior_scale in [0.01, 0.05]:
                for seasonality_mode in ["additive", "multiplicative"]:
                    yield {
                        "changepoint_range": changepoint_range,
                        "seasonality_prior_scale": seasonality_prior_scale,
                        "changepoint_prior_scale": changepoint_prior_scale,
                        "seasonality_mode": seasonality_mode,
                    }


def __measure_error_metric(
    data: Dataset, actual: pd.DataFrame, predicted: np.array
) -> float:
    """Measure the error"""
    # change actual to a numpy array
    actual_this = actual.loc[:, data.subset_column_name].values
    result = np.sqrt(mean_squared_error(y_true=actual_this, y_pred=predicted))
    return result


def __remove_timezone_from_dates(dates_df: pd.DataFrame) -> pd.DataFrame:
    """Remove the timezone from the dates and return a dataframe with a column named ds"""
    dates_no_time_zone = pd.to_datetime(dates_df["ds"]).dt.tz_localize(None)

    return pd.DataFrame({"ds": dates_no_time_zone})


def __score_model(model: Model, data: Dataset) -> float:
    """Score the model on a test set"""
    in_sample_dates = __get_training_dates(data)
    """Removes the time from datetime"""
    in_sample_dates_no_timezone = __remove_timezone_from_dates(in_sample_dates)
    forecast = model.predict(in_sample_dates_no_timezone)
    return __measure_error_metric(
        data, __get_training_set(data), forecast["yhat"].values
    )


def __get_best_model(data: Dataset) -> Tuple[Model, int]:
    """Get the best model based on the configs"""
    best_score = float("inf")
    best_model = None
    seasonality_settings = __get_seasonality_settings(data)
    seasonality_range = __convert_settings_to_seasonality_range(seasonality_settings)

    logging.info(f"Seasonality range: {seasonality_range}")

    configs = __get_configs(seasonality_prior_scale_list=seasonality_range)

    training_df = __get_training_set(data)
    renamed_training_df = training_df.rename(
        columns={"Date": "ds", data.subset_column_name: "y"}
    )
    """Removes the time from datetime"""
    renamed_training_df_notimezone = pd.to_datetime(
        renamed_training_df["ds"]
    ).dt.tz_localize(None)
    renamed_training_df = pd.concat(
        [renamed_training_df_notimezone, renamed_training_df["y"]], axis=1
    )

    best_config = dict()

    # add holidays if needed
    if data.name in ["Indian city pollution"]:
        holiday_df = __add_holidays_for_a_given_country(
            country="India",
            city=data.subset_row_name,
            date_range=[data.values.index.values.min(), data.values.index.values.max()],
        )
        logging.info(
            f"holiday_df for {data.name} subset {data.subset_row_name} : {holiday_df}"
        )
    else:
        holiday_df = None

    for config in list(configs):
        model = None

        logging.info(
            f"Trying config: \n{str(config)} \nfor dataset {data.name} - {data.subset_row_name}"
        )
        try:
            model = __fit_prophet_model(
                data=data,
                training_data=renamed_training_df,
                config=config,
                holiday_df=holiday_df,
            )
        except Exception as e:
            logging.error(f"Error: {e}")
            raise e

        if model is None:
            continue

        score = __score_model(model, data)

        # log score if not nan
        if not np.isnan(score):
            logging.info(f"\n\nScore: {score}")

        if score < best_score:
            logging.info(f"New best score: {score}")
            best_score = score
            best_model = model
            best_config = config

        del model
    # log the best config formatted nicely
    formatted_config = ", ".join(
        [f"{key}={value}" for key, value in best_config.items()]
    )
    logging.info(f"\n\nBest config: {formatted_config} -> {best_score}")
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


def __convert_settings_to_seasonality_range(settings: Dict) -> List:
    """Convert the settings to a list of seasonality ranges"""
    seasonality_range = []
    for time_unit, value in settings.items():
        if value:
            if time_unit == "hours":
                seasonality_range.append(24)
            elif time_unit == "days":
                seasonality_range.append(7)
            elif time_unit == "weeks":
                seasonality_range.append(52)
            elif time_unit == "months":
                seasonality_range.append(12)
            elif time_unit == "years":
                seasonality_range.append(10)
            else:
                raise ValueError(f"Time unit {time_unit} not recognized")

    # add -1 and +1 to the seasonality range
    seasonality_minus_one = [x - 1 for x in seasonality_range]
    seasonality_plus_one = [x + 1 for x in seasonality_range]
    seasonality_range.extend(seasonality_minus_one)
    seasonality_range.extend(seasonality_plus_one)

    return seasonality_range


def __check_if_special_seasonality_is_needed(data: Dataset) -> bool:
    """Check if the data has a special seasonality"""
    if data.name == "Sun spots":
        return True
    return False


def __add_holidays_for_a_given_country(
    country: str,
    date_range: List,
    city: Optional[str] = None,
) -> pd.DataFrame:
    """Add holidays for a given country"""
    country_code = __get_two_letter_country_code(country)
    if city is not None:
        state = __get_state_given_indian_city(city)
        holidays_object = holidays.country_holidays(country_code, subdiv=state)
    else:
        holidays_object = holidays.country_holidays(country_code)
    print(f"Holidays for {country} ({country_code}) - {state}: {holidays_object}")

    holidays_df = pd.DataFrame(columns=["ds", "holiday"])

    # beef up date range to include every day
    date_range_full = pd.date_range(start=date_range[0], end=date_range[-1])
    # add holidays to the date range

    for date in date_range_full:
        if holidays_object.get(date) is not None:
            holidays_df = holidays_df.append(
                {"ds": date, "holiday": holidays_object.get(date)}, ignore_index=True
            )

    print(f"Holidays df: {holidays_df.head()}")
    # drop na any rows with na
    holidays_df = holidays_df.dropna(axis=0, how="any")
    print(f"Holidays df after dropping na: {holidays_df.head()}")
    # drop duplicates
    holidays_df = holidays_df.drop_duplicates(subset=["ds", "holiday"])
    print(f"Holidays df after dropping duplicates: {holidays_df.head()}")

    return holidays_df


def __get_two_letter_country_code(country_full_name: str) -> str:
    """Get the two letter country code"""
    country_code = pycountry.countries.search_fuzzy(country_full_name)[0].alpha_2
    return country_code


def __get_state_given_indian_city(city: str) -> str:
    """Get the state given an Indian city"""
    city_state_abbrev_dict = {
        "Ahmedabad": "GJ",
        "Aizawl": "MZ",
        "Amaravati": "AP",
        "Amritsar": "PB",
        "Bengaluru": "KA",
        "Bhopal": "MP",
        "Brajrajnagar": "OR",
        "Chandigarh": "CH",
        "Chennai": "TN",
        "Coimbatore": "TN",
        "Delhi": "DL",
        "Ernakulam": "KL",
        "Gurugram": "HR",
        "Guwahati": "AS",
        "Hyderabad": "TS",
        "Jaipur": "RJ",
        "Jorapokhar": "JH",
        "Kochi": "KL",
        "Kolkata": "WB",
        "Lucknow": "UP",
        "Mumbai": "MH",
        "Patna": "BR",
        "Shillong": "ML",
        "Talcher": "OR",
        "Thiruvananthapuram": "KL",
        "Visakhapatnam": "AP",
    }
    # handle special cases of missing state abbreviations
    if city not in city_state_abbrev_dict.keys():
        raise ValueError(f"\n\nCity {city} not found in city_state_abbrev_dict - please add it in the function __get_state_given_indian_city")

    return city_state_abbrev_dict[city]


# TODO: Add settings for the model to include seasonality, holidays, etc.
def __fit_prophet_model(
    data: Dataset,
    training_data: pd.DataFrame,
    config: dict,
    holiday_df: pd.DataFrame = None,
) -> Model:
    """Fit a Prophet model to the first 80% of the data"""

    logging.info(f"config being used : {config}")

    # check for seasonality item keys and skip if not needed
    if ("seasonality_prior_scale" not in config.keys()) or (
        "seasonality_mode" not in config.keys()
    ):
        model = Prophet(
            changepoint_range=config["changepoint_range"],
            changepoint_prior_scale=config["changepoint_prior_scale"],
            holidays=holiday_df,
        )

    else:

        model = Prophet(
            seasonality_prior_scale=config["seasonality_prior_scale"],
            changepoint_range=config["changepoint_range"],
            changepoint_prior_scale=config["changepoint_prior_scale"],
            seasonality_mode=config["seasonality_mode"],
            holidays=holiday_df,
        )

        if __check_if_special_seasonality_is_needed(data):
            model.add_seasonality(
                name="cycle_11years",
                period=11,
                fourier_order=5,
                prior_scale=0.1,
                mode="multiplicative",
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

    future_dates_local_df = __remove_timezone_from_dates(__get_future_dates(data))

    in_sample_dates_local_df = __remove_timezone_from_dates(__get_training_dates(data))

    # TODO: Add settings for the model to include holidays, etc.

    in_sample_only_prediction = model.predict(in_sample_dates_local_df)
    out_of_sample_forecast = model.predict(future_dates_local_df)

    forecast_df = out_of_sample_forecast.set_index(keys=["ds"])
    predict_in_sample_df = in_sample_only_prediction.set_index(keys=["ds"])
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
        color=get_color_map_by_method("Prophet"),
        in_sample_prediction=predict_in_sample_df["yhat"],
    )


prophet = method(__get_best_model, __forecast)
