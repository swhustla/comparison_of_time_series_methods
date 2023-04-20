"""Load in electricity consumption data from 
gefcom2017data (This is a dataset of electricity 
demand for 10 zones in the US)."""

from pathlib import Path
from typing import TypeVar, Generator, List
import requests
import pyreadr

from sklearn.preprocessing import StandardScaler

import pandas as pd

Data = TypeVar("Data", contravariant=True)
from data.dataset import Dataset
from data.impute_data import impute

import logging

# set logging level
logging.basicConfig(level=logging.INFO)

PATH_IN_DATA_DIR = Path("data", "gefcom.rda")

def __download_gefcom_data(file_path: Path) -> None:
    """Downloads the GEFCom 2017 dataset."""
    url = "https://github.com/camroach87/gefcom2017data/raw/master/data/gefcom.rda"
    response = requests.get(url, allow_redirects=True)
    
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
    else:
        raise Exception(f"Download failed with status code: {response.status_code}")


__zone_choice = ["CT"]

def __impute_data_if_needed(data: pd.DataFrame) -> pd.DataFrame:
    """Impute the data if any columns have at least 5% missing data."""

    # TODO - cache imputed data
    # TODO - add a check to see if the data is already imputed

    print("Imputing data")
    data = impute(data, target_columns=__zone_choice)

    return data

def __resample(dataframe: Data) -> pd.DataFrame:
    """Resample the data to weekly."""
    return dataframe.resample("w").mean()

def __get_energy_demand_all_zones(scale: bool = True) -> pd.DataFrame:
    """Loads the energy demand dataset and optionally scales the data."""
    try:
        result = pyreadr.read_r(PATH_IN_DATA_DIR)
    except pyreadr.PyreadrError:
        logging.info(f"File not found: {PATH_IN_DATA_DIR}. Downloading the dataset...")
        __download_gefcom_data(PATH_IN_DATA_DIR)
        result = pyreadr.read_r(PATH_IN_DATA_DIR)

    df = result["gefcom"].pivot(index="ts", columns="zone", values="demand")
    df = df.asfreq("D")
    
    if scale:
        scaler = StandardScaler()
        df = pd.DataFrame(data=scaler.fit_transform(df), columns=df.columns, index=df.index)
    
    return df

def energy_demand(zone: list = __zone_choice,
                  scale: bool = True) -> Dataset:
    """Loads the energy demand dataset and optionally scales the data."""
    df = __get_energy_demand_all_zones(scale=scale)
    
    df_this_zone = df[zone]
    df_this_zone = __impute_data_if_needed(df_this_zone)
    df_this_zone = __resample(df_this_zone)
    return Dataset(
        name=f"energy_demand_{zone}",
        values=df_this_zone,
        time_unit="weeks",
        number_columns=[zone],
        subset_row_name=None,
        subset_column_name=zone,
        seasonality=True
        )






