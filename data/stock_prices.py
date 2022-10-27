"""Load in stock prices data.

This is a dataset of stock prices for the company JPMorgan Chase & Co. (JPM).
"""

from datetime import datetime
from typing import TypeVar, Generic, Callable, List, Dict, Any, Generator
from pandas_datareader.data import DataReader

import pandas as pd

Data = TypeVar("Data", contravariant=True)
from data.dataset import Dataset


from .load import Load


__stock_choice = "JPM"


def __load_data(stock_choice=__stock_choice) -> Data:
    """Load in the data."""
    try:
        yahoo_data = DataReader(
            stock_choice, "yahoo", datetime(2001, 6, 1), datetime(2020, 2, 20)
        )
    except ConnectionError as e:
        print("Connection failed")
        raise e

    df = yahoo_data["Adj Close"].to_frame().reset_index("Date")
    df.set_index("Date", inplace=True)
    return df


def __resample(dataframe: Data) -> Data:
    """Resample the data to daily and fill in gaps."""
    yahoo_all_dates_df = dataframe.resample("D").last()
    yahoo_all_dates_df.ffill(inplace=True)
    return yahoo_all_dates_df


def __add_inferred_freq_to_index(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add an inferred frequency to the index."""
    dataframe.index.freq = dataframe.index.inferred_freq
    return dataframe


def stock_prices(stock_choice_list: list = __stock_choice) -> Generator[Dataset, None, None]:
    """Load in the stock prices data."""
    for stock_choice in stock_choice_list:
        path = __load_data(stock_choice)
        data = __resample(path)
        data = __add_inferred_freq_to_index(data)
        yield Dataset(
            "Stock price", data, "days", ["Adj Close"], stock_choice, "Adj Close", False
        )
