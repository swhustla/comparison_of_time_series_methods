"""Load in India Pollution data."""

from pathlib import Path
from typing import TypeVar, Callable, Dict, Tuple, List, Generator, Optional, Union
import pandas as pd

import logging
# set logging level
logging.basicConfig(level=logging.INFO)

from geopy import geocoders

Data = TypeVar("Data", contravariant=True)
from data.dataset import Dataset


from .load import Load

import zipfile


# TODO: add weather data per city
# Boundary layer height and radiation data / cloud cover data (ECMWF) - also have a forecast for this (Blaise has a script for this)
# ECMWF data comes in a netcdf format, so we need to convert it to a csv format
# https://towardsdatascience.com/read-netcdf-data-with-python-901f7ff61648

# Temperature / wind speed / wind direction / humidity (NOAA weather data source)
# Phase II
# Benzene,Toluene,Xylene? (VOCs) -> VOCs are one of the main drivers of PM2.5 in the presence of sunlight (radiation data might be a good predictor of this)

# TODO - add electricity consumption data / electricitymap.org
# TODO - add coal / renewable energy data breakdown
# https://app.electricitymaps.com/zone/IN-DL?utm_source=electricitymaps.com&utm_medium=website&utm_campaign=banner 
# https://www.electricitymaps.com/
# https://github.com/electricitymaps/electricitymaps-contrib#data-sources/blob/master/DATA_SOURCES.md#real-time-electricity-data-sources


def __download_if_needed():
    """Download the data from Kaggle if it is not already downloaded."""

    full_path = Path("rohanrao/air-quality-data-in-india")
    owner_slug = str(full_path).split("/")[0]
    dataset_slug = str(full_path).split("/")[1]
    path = Path(dataset_slug)
    if not path.exists():
        import kaggle

        kaggle.api.datasets_download(owner_slug=owner_slug, dataset_slug=dataset_slug)
        zipfile.ZipFile(f"{dataset_slug}.zip").extractall(dataset_slug)
    else:
        print(f"Path {path} exists")

    return path


def __load_data(path: Path) -> Data:
    """Load in the data from the given path."""
    df = pd.read_csv(path / "city_day.csv", parse_dates=["Date"])
    df.Date = pd.to_datetime(df.Date)

    df.set_index("Date", inplace=True)
    return df


__column_choice = ["PM2.5", "PM10", "O3", "CO", "SO2", "NO2"]
__city_choice = ["Delhi"]


def __preprocess(dataframe: Data) -> pd.DataFrame:
    """Preprocess the data."""
    for column in __column_choice:
        dataframe[column] = dataframe[column].interpolate(method="spline", order=1)
    dataframe.bfill(inplace=True)
    dataframe.dropna(inplace=True, axis=0)
    return dataframe


def __resample(dataframe: Data) -> pd.DataFrame:
    """Resample the data to weekly."""
    return dataframe.resample("w").mean()


def __add_inferred_freq_to_index(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add an inferred frequency to the index."""
    dataframe.index.freq = dataframe.index.inferred_freq
    return dataframe


def get_lat_long_for_city(city: list) -> Dict[str, Tuple[float, float]]:
    """Get the latitude and longitude for the given cities."""
    geolocator = geocoders.Nominatim(user_agent="india-pollution")
    dict_of_lat_long = {}
    for city_this in city:
        location = geolocator.geocode(city_this)
        dict_of_lat_long[city_this] = (location.latitude, location.longitude)
    return dict_of_lat_long


def get_list_of_city_names() -> list:
    """Get a list of city names."""
    logging.info("Getting list of city names")
    path = __download_if_needed()
    data = __load_data(path)
    return list(data.City.unique())


def india_pollution(
    city_list: list = __city_choice,
    pollution_columns: list = __column_choice,
    get_lat_long: bool = False,
) -> Generator[Dataset, None, None]:
    """Load in India Pollution data."""
    path = __download_if_needed()
    data = __load_data(path)

    if get_lat_long:
        dataframe_all_cities = pd.DataFrame()

        lat_long_dict = get_lat_long_for_city(city_list)
        logging.debug(f"Lat long dict: {lat_long_dict}")
        for city_this in city_list:
            lat_long_dataframe = data[data.City == city_this][pollution_columns]
            lat_long_dataframe = __resample(lat_long_dataframe)
            lat_long_dataframe = __add_inferred_freq_to_index(lat_long_dataframe)
            lat_long_dataframe["Latitude"] = lat_long_dict[city_this][0]
            lat_long_dataframe["Longitude"] = lat_long_dict[city_this][1]
            lat_long_dataframe["City"] = city_this

            dataframe_all_cities = pd.concat(
                [dataframe_all_cities, lat_long_dataframe], axis=0
            )
        # store the data to data folder
        logging.info(f"Storing data for {city_list} to data folder")
        dataframe_all_cities.to_csv("data/india_pollution.csv")

    for city in city_list:
        data_this_city = data[data["City"].isin([city])][pollution_columns]
        data_this_city = __preprocess(data_this_city)
        data_this_city = __resample(data_this_city)
        data_this_city = __add_inferred_freq_to_index(data_this_city)
        yield Dataset(
            name="Indian city pollution",
            values= data_this_city,
            time_unit= "weeks",
            number_columns= pollution_columns,
            subset_row_name= city,
            subset_column_name= pollution_columns[0],
            seasonality= True,
        )
