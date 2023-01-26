"""Method to load in the reports from the reports folder."""

from typing import Generator, List, Optional, Tuple, Generic, TypeVar, Protocol

Report = TypeVar("Report")
Data = TypeVar("Data", contravariant=True)

class LoadFromCsv(Protocol[Report]):
    def __call__(self) -> Generator[Report, None, None]:
        ...
