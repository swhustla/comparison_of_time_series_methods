"""Module to load in the reports from the reports folder."""

from pathlib import Path
from typing import Generator, List, Optional, Tuple, Generic, TypeVar, Protocol

import gzip

import pandas as pd

from methods.report_loader import report_loader as method
from methods.json_report_loader import json_report_loader as json_method

from predictions.Prediction import PredictionData

Report = TypeVar("Report", contravariant=True)
Data = TypeVar("Data", contravariant=True)


import json
import gzip
import pandas as pd
from pathlib import Path



def __zipped_json_file_to_prediction_data_object(file_path: Path) -> PredictionData:
    """Method to load in the reports from the reports folder."""
    try:
        with gzip.open(file_path, "rb") as f:
            data = json.load(f)
    except FileNotFoundError as e:
        print(f"The file {file_path} could not be found.")
        raise e
    except Exception as e:
        print(f"An error occurred while loading the file {file_path}: {e}")
        raise e
    
    # the data is a dictionary, we want it to be a PredictionData object

    return PredictionData(**data)


def __csv_file_to_dataframe(file: Path) -> pd.DataFrame:
    """Method to load in the reports from the reports folder."""
    report_df = pd.read_csv(file, index_col=0)
    report_df.index = pd.to_datetime(report_df.index, dayfirst=True)
    return report_df


def __load_reports() -> Optional[Report]:
    """Method to load in the reports from the reports folder."""
    for report in Path("reports").glob("summary_report.csv"):
        report = __csv_file_to_dataframe(report)
        return report



report_loader = method(__load_reports)
json_report_loader = json_method(__zipped_json_file_to_prediction_data_object)