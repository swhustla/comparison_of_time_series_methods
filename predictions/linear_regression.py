import numpy as np
import pandas as pd
from typing import Tuple, Dict

from data.Data import Dataset, Result, Error, Output
from predictions.Prediction import PredictionData

from methods.linear_regression import linear_regression as method


""" 
the linear model is:
`y = beta0 + beta1 * x + epsilon`
for our current problem the model is:
`B= (XT. X)-1. XT. Y` """


def __get_x_matrix(data: Dataset):
    data.reset_index(inplace=True, drop=True)
    x = data.index.values
    return np.vstack((np.ones(len(x)), x)).T


def __get_y_matrix(data: Dataset):
    return data.values


def __get_beta_hat(x, y):
    return np.linalg.inv(x.T @ x) @ x.T @ y


def __predict_using_coefficients(x, coefficients):
    return x.dot(coefficients)


__percent_test = 0.1


def __get_training_data(data: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    number_of_points = int(len(data.values) * (1 - __percent_test))
    return (
        __get_x_matrix(data.values)[:number_of_points],
        __get_y_matrix(data.values)[:number_of_points],
    )


def __get_test_data(data: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    number_of_points = int(len(data.values) * __percent_test)
    return (
        __get_x_matrix(data.values)[-number_of_points:],
        __get_y_matrix(data.values)[-number_of_points:],
    )


def __train(data: Dataset) -> np.ndarray:
    x, y = __get_training_data(data)
    return __get_beta_hat(x, y)


def __convert_to_pandas_series(data: np.ndarray) -> pd.Series:
    return pd.Series(data=data[:, 0], index=range(len(data)))


def __test(data: Dataset, theta) -> Tuple[PredictionData, Dataset]:
    x, y = __get_test_data(data)
    prediction = __predict_using_coefficients(x, theta)
    title = f"{data.subset_column_name} forecast for {data.subset_row_name} with Linear Regression"
    return PredictionData(
        __convert_to_pandas_series(prediction), None, title
    ), Dataset(__convert_to_pandas_series(y), None, None, None, None)


linear_regression = method(__train, __test)
