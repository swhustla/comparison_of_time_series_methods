"""
Tsetlin Machine Regression method

This module contains the Tsetlin Machine Regression method.
It is a wrapper around the pyTsetlinMachine library.

"""

from typing import Tuple, TypeVar, Callable, Dict
import pandas as pd
from data.dataset import Dataset
from predictions.Prediction import PredictionData
import logging

from .predict import Predict

Data = TypeVar("Data", contravariant=True)
Model = TypeVar("Model")
Prediction = TypeVar("Prediction")

def tsetlin_machine(
    seasonal_decompose_data: Callable[[Dataset], Dataset],
    get_best_model_config: Callable[[Dataset], Dict],
    get_forecast: Callable[[Dataset, dict], Prediction],
    combine_trend_seasonal_residual: Callable[[Prediction, Prediction, Prediction, Data, Data], Prediction],
) -> Predict[Dataset, PredictionData]:
    """
    Return a function that takes a dataset and returns a prediction.
    """
    def predict(dataset: Dataset, parallel: bool = False) -> PredictionData:
        """
        Return a prediction for the given dataset.
        In this case, the prediction is a combination of the trend, seasonal and residual predictions.
        
        """
        logging.info("Tsetlin Machine Regression")
        # make a copy of the dataset by iterating through the object
        dataset = Dataset(**dataset.__dict__)


        # decompose the data into trend, seasonal and residual components
        stl_dataset = seasonal_decompose_data(dataset)
        trend_dataset_values = stl_dataset.values.trend
        seasonal_dataset_values = stl_dataset.values.seasonal
        residual_dataset_values = stl_dataset.values.resid

        # print out the three components in three columns for easy comparison
        input_data_components = pd.concat(
        [trend_dataset_values, seasonal_dataset_values, residual_dataset_values], axis=1
        )
        input_data_components.columns = ["trend", "seasonal", "residual"]
        logging.info(f"Three input components:\n{input_data_components.head()}")

        list_of_predictions = []
        for dataset_component in [trend_dataset_values, seasonal_dataset_values, residual_dataset_values]:
            dataset.values = dataset_component
            logging.info(f"Running Tsetlin for stream: {dataset_component.name}")
            params = get_best_model_config(dataset, parallel)   
            list_of_predictions.append(get_forecast(dataset, params))
        return combine_trend_seasonal_residual(*list_of_predictions, stl_dataset, dataset)
            
    return predict

def tsetlin_machine_single(
    get_best_model_config: Callable[[Dataset], Dict],
    get_forecast: Callable[[Dataset, dict], Prediction],
) -> Predict[Dataset, PredictionData]:
    """
    Return a function that takes a dataset and returns a prediction.
    """
    def predict(dataset: Dataset, parallel: bool = False) -> PredictionData:
        """
        Return a prediction for the given dataset.
        """
        logging.info("Tsetlin Machine Regression")
        params = get_best_model_config(dataset, parallel)
        return get_forecast(dataset, params)
    return predict




