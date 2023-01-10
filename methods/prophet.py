""" Facebook Prophet method for time series forecasting. 

    This module contains the Prophet method for time series forecasting.
    
    It is a wrapper around the Prophet library.
"""

from typing import TypeVar, Callable, Any, Dict, List, Optional, Tuple, Type, Union, Generator
from data.dataset import Dataset
from predictions.Prediction import PredictionData

Data = TypeVar("Data", contravariant=True)
Model = TypeVar("Model")
Prediction = TypeVar("Prediction")

from .predict import Predict


def prophet(
    get_best_model: Callable[[Dataset], Tuple[Model, int]],
    forecast: Callable[[Model, Dataset], PredictionData],
) -> Predict[Dataset, PredictionData]:
    def predict(
        data: Dataset,
    ) -> Prediction:
        trained_model, number_of_configs = get_best_model(data)
        return forecast(trained_model, data, number_of_configs)

    return predict
    