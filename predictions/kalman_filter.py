"""
The Kalman Filter is a state space model that is used to estimate the state of a system.

THis method uses the pykalman library to fit a Kalman Filter model to the data.


"""

from typing import TypeVar
from methods.kalman_filter import kalman_filter as method
import pandas as pd

import logging

from pykalman import KalmanFilter

from data.dataset import Dataset, Result
from predictions.Prediction import PredictionData

Model = TypeVar("Model")


def __number_of_steps(data: Dataset) -> int:
    return int(len(data.values) // 5)

def __get_training_set(data: Dataset) -> pd.DataFrame:
    return data.values[: -__number_of_steps(data)][data.subset_column_name]

def __get_test_set(data: Dataset) -> pd.DataFrame:
    return data.values[-__number_of_steps(data) :][data.subset_column_name]

def __fit_kalman_filter_model(data: Dataset) -> Model:
    """Fit a Kalman Filter model to the data"""

    kf = KalmanFilter(
        initial_state_mean=0,
        n_dim_obs=1,
        em_vars=["transition_matrices", "observation_matrices"],
    )
    kf = kf.em(__get_training_set(data), n_iter=5)
    return kf

def __forecast(model: Model, data: Dataset) -> PredictionData:
    """Forecast the next 20% of the data"""
    return PredictionData(
        values=model.smooth(__get_test_set(data))[0],
        prediction_column_name=data.subset_column_name,
        ground_truth_values=__get_test_set(data),
        confidence_columns=None,
        title=f"{data.subset_column_name} forecast for {data.subset_row_name} with Kalman Filter",
    )
