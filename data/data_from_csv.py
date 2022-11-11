"""Load in a new dataset from a csv"""

import pandas as pd 

from pathlib import Path
from data.dataset import Dataset

__csv_path = Path("data/my_csv.csv")

def __add_inferred_freq_to_index(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add an inferred frequency to the index."""
    dataframe.index.freq = dataframe.index.inferred_freq
    return dataframe


def __load_data_if_needed() -> Dataset:
    """Load in the csv data if needed."""
    if not __csv_path.exists():
        raise FileNotFoundError("The csv file does not exist")
    # if date column is called "ds" then it will be parsed as a date
    try:
        data = pd.read_csv(__csv_path, parse_dates=["ds"]).set_index("ds")
        data.index.name = "Date"
    except ValueError:
        data = pd.read_csv(__csv_path).set_index("Date")

    return __add_inferred_freq_to_index(data)


def load_from_csv() -> Dataset:
    """Load in the csv data."""
    return Dataset(
        name="My csv",
        values=__load_data_if_needed(),
        time_unit="days",
        number_columns=["y"],
        subset_row_name="all",
        subset_column_name="y",
        seasonality=True,
    )
        

