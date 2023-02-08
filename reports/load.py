"""Method to load in the reports from the reports folder."""

from typing import Generator, List, Optional, Tuple, Generic, TypeVar, Protocol

from predictions.Prediction import PredictionData


Report = TypeVar("Report")
Data = TypeVar("Data", contravariant=True)
String = TypeVar("String")


class LoadFromCsv(Protocol[Report]):
    def __call__(self) -> Generator[Report, None, None]:
        ...


class LoadFromJson(Protocol[String]):
    def __call__(self, file_path: String) -> Data:
        ...
