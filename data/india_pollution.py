"""Load in India Pollution data."""

from pathlib import Path
from typing import TypeVar, Callable
import pandas as pd

Data = TypeVar("Data", contravariant=True)
from data.dataset import Dataset


from .load import Load

import zipfile


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


def __preprocess(dataframe: Data) -> Data:
    """Preprocess the data."""
    for column in __column_choice:
        dataframe[column] = dataframe[column].interpolate(method="spline", order=1)
    dataframe.bfill(inplace=True)
    dataframe.dropna(inplace=True, axis=0)
    return dataframe


def __resample(dataframe: Data) -> Data:
    """Resample the data to weekly."""
    return dataframe.resample("w").mean()

def __add_inferred_freq_to_index(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add an inferred frequency to the index."""
    dataframe.index.freq = dataframe.index.inferred_freq
    return dataframe

def india_pollution(city: list = __city_choice, pollution_columns : list = __column_choice) -> Dataset:
    """Load in India Pollution data."""
    path = __download_if_needed()
    data = __load_data(path)
    data = data[data["City"].isin(city)][pollution_columns]
    data = __preprocess(data)
    data = __resample(data)
    data = __add_inferred_freq_to_index(data)
    return Dataset("Indian city pollution", data, "weeks", pollution_columns, city[0], pollution_columns[0], True)
