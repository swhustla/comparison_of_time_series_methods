"""Impute data using a given method."""

import pandas as pd
import logging
from data.dataset import Dataset



def impute(dataframe: pd.DataFrame, target_columns = list) -> pd.DataFrame:
    """Impute the data using the given method."""
    logging.info(f"Imputing data for columns {target_columns}")
    dataframe = dataframe.copy()
    for column in target_columns:
        if dataframe[column].isna().sum() > 0:
            logging.info(f"Imputing {column}")
            dataframe[column].interpolate(method="time", inplace=True)
            dataframe[column].fillna(method="bfill", inplace=True)
    return dataframe