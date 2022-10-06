"""Load in sun spots data.

This is a dataset of the number of sun spots observed from 1700 to 2008.
"""

from typing import TypeVar
import statsmodels.api as sm
import pandas as pd


Data = TypeVar("Data", contravariant=True)
from data.dataset import Dataset


from .load import Load


def sun_spots() -> Data:
    """Load in and prepare the data."""
    data = sm.datasets.sunspots.load_pandas().data
    data.index = pd.Index(pd.date_range("1700", end="2009", freq="A-DEC"))
    data.index.name = "Date"
    del data["YEAR"]

    return Dataset(
        name="Sun spots",
        values=data,
        time_unit="years",
        number_columns=["SUNACTIVITY"],
        subset_row_name="All",
        subset_column_name="SUNACTIVITY",
        seasonality=True,
    )

