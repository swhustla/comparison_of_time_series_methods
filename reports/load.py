"""Method to load in the reports from the reports folder."""

from typing import Generator, List, Optional, Tuple, Generic, TypeVar, Protocol

from data.report import Report
Data = TypeVar("Data", contravariant=True)

class LoadReport(Protocol[Report], Generic[Data]):
    def __call__(self) -> Generator[Report, None, None]:
        pass

