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
from data.impute_data import impute


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

import sys
__parent_folder = Path(sys.argv[0] if __name__ == "__main__" else "").resolve().parent
__destination_folder = Path(__parent_folder) / "data"


def __download_if_needed():
    """Download the data from Kaggle if it is not already downloaded."""

    full_path = Path("rohanrao/air-quality-data-in-india")
    owner_slug = str(full_path).split("/")[0]
    dataset_slug = str(full_path).split("/")[1]

    local_data_path = __destination_folder / dataset_slug

    if not local_data_path.exists():
        logging.info("Downloading data from Kaggle")
        import kaggle

        kaggle.api.datasets_download(owner_slug=owner_slug, dataset_slug=dataset_slug)
        zipfile.ZipFile(f"{dataset_slug}.zip").extractall(path=local_data_path)
    else:
        logging.info(f"Path {local_data_path} exists")

    return local_data_path


def __load_data(path: Path) -> Data:
    """Load in the data from the given path."""
    df = pd.read_csv(path / "city_day.csv", parse_dates=["Date"])
    df.Date = pd.to_datetime(df.Date)

    df.set_index("Date", inplace=True)
    return df


__column_choice = ["PM2.5", "PM10", "O3", "CO", "SO2", "NO2"]
__city_choice = ["Delhi"]


def __impute_data_if_needed(data: pd.DataFrame) -> pd.DataFrame:
    """Impute the data if any columns have at least 5% missing data."""

    # TODO - cache imputed data
    # TODO - add a check to see if the data is already imputed

    print("Imputing data")
    data = impute(data, target_columns=__column_choice)

    return data


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


__list_of_coastal_cities = ["Mumbai", "Chennai", "Kolkata", "Visakhapatnam", "Goa", "Pondicherry"]

__list_of_inland_cities = ["Bhopal", "Delhi", "Hyderabad", "Jaipur", "Lucknow", "Patna", "Ranchi", "Srinagar", "Thiruvananthapuram", "Bengaluru"]

def get_list_of_coastal_indian_cities() -> list:
    """Get a list of coastal cities in India, given the list of cities."""
    list_of_cities = get_list_of_city_names()
    coastal_cities = []
    for city in list_of_cities:
        if city in __list_of_coastal_cities:
            coastal_cities.append(city)
    return coastal_cities

def get_list_of_inland_indian_cities() -> list:
    """Get a list of inland cities in India, given the list of cities."""
    list_of_cities = get_list_of_city_names()
    inland_cities = []
    for city in list_of_cities:
        if city in __list_of_inland_cities:
            inland_cities.append(city)
    return inland_cities

def get_city_list_by_tier(city_type: str) -> list:
    """Get the list of cities in each tier"""
    tier1 = ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
    tier2 = ["Agra", "Ahmedabad", "Amritsar", "Bhopal", "Coimbatore", "Indore", "Jaipur", "Kanpur", "Lucknow", "Nagpur", "Patna", "Pune", "Surat", "Visakhapatnam"]
    tier3 = ["Aizawl", "Amaravati", "Brajrajnagar", "Chandigarh", "Ernakulam", "Guwahati", "Gurugram", "Jorapokhar", "Kochi", "Shillong", "Talcher", "Thiruvananthapuram"]

    # Handle likely typos
    if city_type == "Tier 1 Cities" or city_type == "Tier 1 cities" or city_type == "Tier 1 city" or city_type == "Tier 1 City" or city_type == "Tier 1":
        return tier1
    elif city_type == "Tier 2 Cities" or city_type == "Tier 2 cities" or city_type == "Tier 2 city" or city_type == "Tier 2 City" or city_type == "Tier 2":
        return tier2
    elif city_type == "Tier 3 Cities" or city_type == "Tier 3 cities" or city_type == "Tier 3 city" or city_type == "Tier 3 City" or city_type == "Tier 3":
        return tier3


__list_of_indo_gangetic_plain_cities = ["Ahmedabad",  "Amritsar", "Delhi", "Gurugram", "Lucknow", "Jaipur", "Patna", "Chandigarh", "Kolkata","Guwahati", "Shillong", "Jorapokhar", "Aizawl"]
__list_of_south_eastern_cities = ["Chennai", "Coimbatore","Amaravati","Bengaluru", "Hyderabad"]
__list_of_western_cities = ["Kochi", "Ernakulam", "Thiruvananthapuram","Mumbai"]
__list_of_north_eastern_cities = ["Bhopal", "Brajrajnagar", "Talcher", "Visakhapatnam",]


def get_cities_from_geographical_region(geographical_region: str)-> list:
    """Get the list of cities in each geographical region"""
    if geographical_region == "Indo-Gangetic Plain" or geographical_region == "Indo Gangetic Plain" or geographical_region == "Northern":
        return __list_of_indo_gangetic_plain_cities
    elif geographical_region == "South Eastern" or geographical_region == "Southern":
        return __list_of_south_eastern_cities
    elif geographical_region == "North Eastern" or geographical_region == "Eastern":
        return __list_of_north_eastern_cities
    elif geographical_region == "Western" or geographical_region == "Western Coast" or geographical_region == "South Western":
        return __list_of_western_cities


def india_pollution(
    city_list: list = __city_choice,
    pollution_columns: list = __column_choice,
    get_lat_long: bool = False,
    raw: bool = False,
) -> Generator[Dataset, None, None]:
    """Load in India Pollution data."""
    path = __download_if_needed()
    data = __load_data(path)

    list_of_city_names = get_list_of_city_names()

    if get_lat_long:
        dataframe_all_cities = pd.DataFrame()

        lat_long_dict = get_lat_long_for_city(city_list)
        logging.debug(f"Lat long dict: {lat_long_dict}")
        for city_this in city_list:
            if city_this not in list_of_city_names:
                raise ValueError(
                    f"City {city_this} not in list of city names: {list_of_city_names}"
                )
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
        if city not in list_of_city_names:
            raise ValueError(
                f"City {city} not in list of city names: {list_of_city_names}"
            )
        data_this_city = data[data["City"].isin([city])][pollution_columns]
        if not raw:
            data_this_city = __impute_data_if_needed(data_this_city)
            data_this_city = __resample(data_this_city)
        data_this_city = __add_inferred_freq_to_index(data_this_city)
        yield Dataset(
            name="Indian city pollution",
            values=data_this_city,
            time_unit="weeks",
            number_columns=pollution_columns,
            subset_row_name=city,
            subset_column_name=pollution_columns[0],
            seasonality=True,
        )
