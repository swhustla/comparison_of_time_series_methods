"""Load in airline stats data."""

import pandas as pd

from pathlib import Path
from data.Data import Dataset

__airline_passenger_path = Path("data/airline-passengers.csv")
__url_airline_passengers = "https://raw.githubusercontent.com/benman1/Machine-Learning-for-Time-Series-with-Python/main/chapter10/passengers.csv"


def __load_data_if_needed() -> pd.DataFrame:
    if not __airline_passenger_path.exists():
        data = pd.read_csv(__url_airline_passengers, parse_dates=["date"]).set_index(
            "date"
        )
        data.to_csv(__airline_passenger_path, index=True)
    return pd.read_csv(__airline_passenger_path, parse_dates=["date"]).set_index("date")


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
