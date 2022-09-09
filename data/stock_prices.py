"""Load in stock prices data."""

from datetime import datetime
from typing import TypeVar
from pandas_datareader.data import DataReader

Data = TypeVar("Data", contravariant=True)
from data.Data import Dataset


from .load import Load




__stock_choice = "JPM"

def __load_data(stock_choice=__stock_choice) -> Data:
    """Load in the data."""
    try:
        yahoo_data = DataReader(stock_choice, "yahoo", datetime(2001, 6, 1), datetime(2020, 2, 20))
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


def stock_prices(stock_choice=__stock_choice) -> Dataset:
    """Load in stock market data."""
    data = __load_data(stock_choice)
    data = __resample(data)
    return Dataset("Stock price", data, "days", ["Adj Close"], "JPM", "Adj Close")
