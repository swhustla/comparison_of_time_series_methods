"""Load in airline stats data."""

import pandas as pd

from pathlib import Path
from data.dataset import Dataset

from data.impute_data import impute

import logging

__airline_passenger_path = Path("data/airline-passengers.csv")
__url_airline_passengers = "https://raw.githubusercontent.com/benman1/Machine-Learning-for-Time-Series-with-Python/main/chapter10/passengers.csv"


def __add_inferred_freq_to_index(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add an inferred frequency to the index."""
    dataframe.index.freq = dataframe.index.inferred_freq
    return dataframe


def __impute_data_if_needed(data: pd.DataFrame) -> pd.DataFrame:
    """Impute the data if needed."""
    
    if data.isnull().values.any():
        print("Imputing data")
        data = impute(data, ["passengers"])

    return data


def __load_data_if_needed() -> pd.DataFrame:
    if not __airline_passenger_path.exists():
        data = pd.read_csv(__url_airline_passengers, parse_dates=["date"]).set_index(
            "date"
        )
        data.to_csv(__airline_passenger_path, index=True)

    data = pd.read_csv(__airline_passenger_path, parse_dates=["date"]).set_index("date")

    return __impute_data_if_needed(__add_inferred_freq_to_index(data))


def airline_passengers() -> pd.DataFrame:
    """Load in the airline passenger data."""
    return Dataset(
        "Airline passengers",
        __load_data_if_needed(),
        "months",
        ["passengers"],
        "all",
        "passengers",
        True,
    )
