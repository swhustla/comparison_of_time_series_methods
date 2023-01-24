"""Apply STL decomposition to the data.

Seasonal decomposition is a time series analysis technique that
decomposes a time series into three components: trend, seasonality,
and noise. This is useful for identifying trends and seasonality in
the data, and for removing them to get a better understanding of the
noise in the data.

This function uses the statsmodels STL decomposition function to
perform the decomposition. The function is applied to each column of
the data, and the results are stored in a new dataset.

"""


import logging
import pandas as pd
from statsmodels.tsa.seasonal import STL
from data.dataset import Dataset

from methods.seasonal_decompose import seasonal_decompose as method

def __seasonal_decompose_data(data: Dataset) -> Dataset:
    """Perform STL decomposition on the data."""
    logging.info("Performing STL decomposition")
    decomposed_data = data.copy()
    for column in data.values.columns:
        logging.info(f"Decomposing column {column}")
        stl = STL(data.values[column], period=12)
        res = stl.fit()
        decomposed_data.values[column] = res.resid
    return decomposed_data


seasonal_decompose = method(__seasonal_decompose_data)