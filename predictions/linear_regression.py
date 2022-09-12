import numpy as np
import pandas as pd
from typing import Tuple

from data.Data import Dataset
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


__percent_test = 0.2


def __get_training_data(data: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    print(type(data))
    print(data)
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
        values=__convert_to_pandas_series(prediction),
        prediction_column_name=None,
        ground_truth_values=__convert_to_pandas_series(y),
        confidence_columns=None,
        title=title,
    )


linear_regression = method(__train, __test)
