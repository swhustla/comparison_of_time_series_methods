"""Load in stock prices data.

This is a dataset of stock prices for the company JPMorgan Chase & Co. (JPM).
"""

import datetime
from typing import TypeVar, Generic, Callable, List, Dict, Any, Generator

import logging

import yfinance as yf

import pandas as pd

Data = TypeVar("Data", contravariant=True)
from data.dataset import Dataset


from .load import Load


__stock_choice = "JPM"


def __load_data(stock_choice=__stock_choice) -> Data:
    """Load in the data."""
    logging.info(f"Loading data for {stock_choice} stock price")
    try:
        # use yfinance to download the data
        ticker = yf.Ticker(stock_choice)
        start = datetime.datetime(2014, 12, 1)
        end = datetime.datetime(2022, 12, 1)
        yahoo_data = ticker.history(start=start, end=end)

    except ConnectionError as e:
        print("Connection failed")
        raise e
    df = yahoo_data["Close"].to_frame().reset_index("Date")
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

def get_a_list_of_value_stock_tickers() -> List[str]:
    """Return a list of value stock tickers.
    these are stocks often considered to be undervalued.
    They are also old companies with a long history of paying dividends.
    Examples include:
    - J.P. Morgan Chase & Co. (JPM)
    - The Coca-Cola Company (KO)
    - The Procter & Gamble Company (PG)
    - The Walt Disney Company (DIS)
    - The Home Depot, Inc. (HD)    
    """

    return ["JPM", "KO", "PG", "DIS", "HD"]


def get_a_list_of_growth_stock_tickers() -> List[str]:
    """Return a list of growth stock tickers.
    These are stocks often considered to be overvalued.
    They are also young companies with a short history of paying dividends.
    Examples include:
    - Apple Inc. (AAPL)
    - Microsoft Corporation (MSFT)
    - Amazon.com, Inc. (AMZN)
    - Facebook, Inc. (FB)
    - Alphabet Inc. (GOOG)
    """

    return ["AAPL", "MSFT", "AMZN", "FB", "GOOG"]



def stock_prices(
    stock_choice_list: list = __stock_choice,
) -> Generator[Dataset, None, None]:
    """Load in the stock prices data."""
    for stock_choice in stock_choice_list:
        path = __load_data(stock_choice)
        data = __resample(path)
        data = __add_inferred_freq_to_index(data)
        yield Dataset(
            name="Stock price",
            values=data,
            time_unit="days",
            number_columns=["Close"],
            subset_row_name=stock_choice,
            subset_column_name="Close",
            seasonality=True,
        )
