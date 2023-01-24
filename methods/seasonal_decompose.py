"""Perform seasonal decomposition using STL decomposition."""

from typing import Callable, TypeVar
from data.dataset import Dataset
from predictions.Prediction import PredictionData

Data = TypeVar("Data", contravariant=True)
Model = TypeVar("Model")
Prediction = TypeVar("Prediction")

from .decompose import Decompose


def seasonal_decompose(
    seasonal_decompose_data: Callable[[Dataset], Dataset],
) -> Decompose[Dataset]:
    def decompose(
        data: Dataset,
    ) -> Dataset:
        decomposed_dataset = seasonal_decompose_data(data)
        return decomposed_dataset

    return decompose