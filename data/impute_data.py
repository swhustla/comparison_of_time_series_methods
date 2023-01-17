"""Impute data using a given method.

Imputation is the process of filling in missing data. This is necessary because many of the methods we use to predict the data require a complete dataset.

Pandas interpolate has been used as the default imputation method. We experimented with a more advanced imputation method, miceforest, but it was not as effective as pandas interpolate.

MICEForest is a package that uses multiple imputation to impute missing values. It is based on the mice package, but uses random forests to impute the data. It is more effective than pandas interpolate, but it is also much slower. It is therefore only used when the data has more than 2 columns and more than 5% of the data is missing.

The multiple imputation method is based on the following paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/
It leverages the fact that with pollution data, there are multiple variables that are correlated. For example, the PM2.5 and PM10 levels are correlated. This means that if we impute the PM2.5 levels, we can use the PM10 levels to impute the missing PM2.5 levels. This is done by using the miceforest package to impute the data multiple times, and then averaging the results.
"""

import pandas as pd
import logging
from data.dataset import Dataset

try:
    import miceforest as mf
except Exception as e:
    logging.error(f"Error importing miceforest: {e}")


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


def impute(dataframe: pd.DataFrame, target_columns:list) -> pd.DataFrame:
    """Impute the data using the given method."""
    logging.info(f"Imputing data for columns {target_columns}")
    dataframe = dataframe.copy()
    logging.info(f"columns present: {dataframe.columns}")
    logging.info(f"Shape of dataframe before imputation: {dataframe.shape}")
    need_to_impute = False
    advanced_imputation = False
    for column in target_columns:
        number_of_missing_values = dataframe[column].isna().sum()
        if number_of_missing_values > 0:
            need_to_impute = True
            proportion_missing = number_of_missing_values / len(dataframe[column])
            if proportion_missing > 0.05 and proportion_missing < 0.50 and (len(target_columns) > 2):
                logging.info(f"Column {column} has {number_of_missing_values}; more than 5% missing values, less than 50%, so advanced imputation is necessary")
                advanced_imputation = True
            else:
                logging.info(f"Column {column} has {number_of_missing_values}; less than 5% missing values, so only simple imputation is necessary")

    if need_to_impute:
        logging.info("Imputation is necessary")
        if advanced_imputation: # advanced imputation if data is not seasonal
            dataframe[target_columns] = __impute_with_miceforest(dataframe[target_columns])
        else: # simple imputation
            dataframe[target_columns] = __impute_with_pandas_interpolate(dataframe[target_columns])

        logging.info(f"Shape of dataframe after imputation: {dataframe.shape}")

    return dataframe