"""Impute data using a given method."""

import pandas as pd
import logging
from data.dataset import Dataset

import miceforest as mf


def __impute_with_pandas_interpolate(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Use the pandas interpolate method to impute the data."""
    logging.info("Imputing data with pandas interpolate")
    dataframe = dataframe.copy()
    dataframe.interpolate(method="time", inplace=True)
    dataframe.fillna(method="bfill", inplace=True)
    return dataframe
    

def __impute_with_miceforest(data: pd.DataFrame) -> pd.DataFrame:
    """Use the miceforest package to impute the data."""
    logging.info("Imputing data with miceforest")
    data = data.copy()
    try:
        kernel = mf.ImputationKernel(
            data=data,
            save_all_iterations=True,
            random_state=1234,
        )
        kernel.mice(5)
        return kernel.impute_new_data(data).complete_data(0)
    except Exception as e:
        logging.error(f"Error imputing data with miceforest: {e}, reverting to pandas interpolate")
        return __impute_with_pandas_interpolate(data)


def impute(dataframe: pd.DataFrame, target_columns = list) -> pd.DataFrame:
    """Impute the data using the given method."""
    logging.info(f"Imputing data for columns {target_columns}")
    dataframe = dataframe.copy()

    need_to_impute = False
    for column in target_columns:
        if dataframe[column].isna().sum() / len(dataframe[column]) > 0.05:
            need_to_impute = True


    if len(target_columns)> 2 and need_to_impute: # advanced imputation
        dataframe[target_columns] = __impute_with_miceforest(dataframe[target_columns])
    else: # simple imputation
        dataframe[target_columns] = __impute_with_pandas_interpolate(dataframe[target_columns])

    return dataframe