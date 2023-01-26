"""Module to load in the reports from the reports folder."""

from pathlib import Path
from typing import Generator, List, Optional, Tuple, Generic, TypeVar, Protocol

import pandas as pd

from methods.report_loader import report_loader as method

Report = TypeVar("Report", contravariant=True)
Data = TypeVar("Data", contravariant=True)


def __file_to_dataframe(file: Path) -> pd.DataFrame:
    """Method to load in the reports from the reports folder."""
    report_df = pd.read_csv(file, index_col=0)
    report_df["Start Time"] = pd.to_datetime(report_df["Start Time"],dayfirst=True)
    return report_df


def __load_reports() -> Optional[Report]:
    """Method to load in the reports from the reports folder."""
    for report in Path("reports").glob("summary_report.csv"):
        report = __file_to_dataframe(report)
        return report



report_loader = method(__load_reports)