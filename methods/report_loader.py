"""Module to load in the reports from the reports folder."""

from pathlib import Path
from typing import Generator, List, Optional, Tuple, Generic, TypeVar, Protocol

from reports.load import LoadReport

Report = TypeVar("Report", contravariant=True)
Data = TypeVar("Data", contravariant=True)


def report_loader(
    load_reports: LoadReport[Report],
) -> Generator[Report, None, None]:
    """Method to load in the reports from the reports folder."""
    def load() -> Generator[Report, None, None]:
        return load_reports()

    return load

