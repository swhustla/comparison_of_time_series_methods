"""Generate a list of tuples for a simple linear regression."""

from typing import TypeVar
import numpy as np
import pandas as pd

Data = TypeVar("Data", contravariant=True)
from data.Data import Dataset

from .load import Load

__chosen_length = 1096


def __increasing_value_with_random_noise(count: int = __chosen_length) -> np.ndarray:
    """Generate an increasing value with random noise."""
    return np.arange(count) + np.random.randint(-10, 10, count)


def __convert_index_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the index to datetime."""
    df.index = pd.to_datetime(df.index, unit="D")
    return df


def __set_index_name(df: pd.DataFrame) -> pd.DataFrame:
    """Set the index name."""
    df.index.name = "Date"
    return df


def __build_data(count: int) -> Data:
    """Build the data."""
    return __convert_index_to_datetime(
        __set_index_name(
            pd.DataFrame({"y": __increasing_value_with_random_noise(count)})
        )
    )


def list_of_tuples(count: int = __chosen_length) -> Dataset:
    """Generate a list of tuples for a simple linear regression."""

    return Dataset("Straight line", __build_data(count), "days", ["y"], "random", "y")
