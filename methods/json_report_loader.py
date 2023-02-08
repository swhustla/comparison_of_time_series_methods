"""Module to load in prediction data from archived json files in the reports/full_data folder."""

from pathlib import Path

from typing import Generator, List, Optional, Tuple, Generic, TypeVar, Protocol

from reports.load import LoadFromJson

from predictions.Prediction import PredictionData

Report = TypeVar("Report", contravariant=True)
Data = TypeVar("Data", contravariant=True)
String = TypeVar("String")


def json_report_loader(
    json_to_prediction_data_object: LoadFromJson[String],
) -> Generator[Report, None, None]:
    """Method to load in the reports from the reports folder."""

    def load(file_path: String) -> Generator[Report, None, None]:
        return json_to_prediction_data_object(file_path=file_path)

    return load
