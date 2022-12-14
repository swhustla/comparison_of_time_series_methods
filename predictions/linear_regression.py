""" 
Linear Regression model

This model uses linear regression to predict the future values of a given dataset. 
This technique was first used by Sir Francis Galton in 1886 to predict the height of a 
child based on the height of their parents.

It is based on the idea that there is a linear relationship between the independent variable
and the dependent variable. This relationship is expressed as a linear equation, which can
be used to predict the value of the dependent variable given the value of the independent
variable.

It is the simplest model that we have implemented, and it is used as a baseline for 
the more complex models.

It's main advantage is that it is very easy to implement, and it is very fast to train.

The negative side of this model is that it is very limited in its ability to predict 
the future values of a time series. It is also very sensitive to outliers.

The training data is the first 80% of the dataset, and the test data is the last 20%.

The linear model used in this implementation is the Ordinary Least Squares model:
https://en.wikipedia.org/wiki/Ordinary_least_squares

`y = beta0 + beta1 * x + epsilon`
for our current problem the model is:
`B= (XT. X)-1. XT. Y` 


"""


import numpy as np
import pandas as pd
from typing import Tuple

from data.dataset import Dataset
from predictions.Prediction import PredictionData

from methods.linear_regression import linear_regression as method


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
    data_this = data.values.copy()
    number_of_points = int(len(data_this) * (1 - __percent_test))
    return (
        __get_x_matrix(data_this)[:number_of_points],
        __get_y_matrix(data_this)[:number_of_points],
    )


def __get_test_data(data: Dataset) -> Tuple[np.ndarray, np.ndarray]:
    data_this = data.values.copy()
    number_of_points = int(len(data_this) * __percent_test)
    return (
        __get_x_matrix(data_this)[-number_of_points:],
        __get_y_matrix(data_this)[-number_of_points:],
    )


def __train(data: Dataset) -> np.ndarray:
    x, y = __get_training_data(data)
    return __get_beta_hat(x, y)


def __revert_index_to_date(data: Dataset, prediction: PredictionData) -> PredictionData:
    prediction.values.index = data.values.index[prediction.values.index]
    prediction.ground_truth_values.index = data.values.index[
        prediction.ground_truth_values.index
    ]
    return prediction


def __convert_to_pandas_series(data: np.ndarray, start_value: int = 0) -> pd.Series:
    return pd.Series(data=data[:, 0], index=range(start_value, start_value + len(data)))


def __test(data: Dataset, theta) -> Tuple[PredictionData, Dataset]:
    x, y = __get_test_data(data)
    prediction = __predict_using_coefficients(x, theta)
    x_insample, y_insample = __get_training_data(data)
    in_sample = __predict_using_coefficients(x_insample, theta)
    title = f"{data.subset_column_name} forecast for {data.subset_row_name} with Linear Regression"
    start_value = len(data.values) - len(prediction)
    in_sample_prediction = __convert_to_pandas_series(in_sample)
    forecast = __convert_to_pandas_series(prediction, start_value=start_value)
    in_sample_prediction = pd.concat([in_sample_prediction,forecast])[:len(x_insample)+2]
    print(f"in_sample_prediction\n{in_sample_prediction}")
    return __revert_index_to_date(
        data=data,
        prediction=PredictionData(
            method_name="Linear Regression",
            values=__convert_to_pandas_series(prediction, start_value=start_value),
            prediction_column_name=None,
            ground_truth_values=__convert_to_pandas_series(y, start_value=start_value),
            confidence_columns=None,
            title=title,
            plot_folder=f"{data.name}/{data.subset_row_name}/linear_regression/",
            plot_file_name=f"{data.subset_column_name}_forecast",
            color="green",
            in_sample_prediction=in_sample_prediction.set_axis(data.values.index[:len(x_insample)+2]),
        ),
    )


linear_regression = method(__train, __test)
