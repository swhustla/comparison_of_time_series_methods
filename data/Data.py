from dataclasses import dataclass
from typing import TypeVar

Data = TypeVar("Data")
Output = list()
Result = TypeVar("Result")
Error = TypeVar("Error")


@dataclass
class Dataset:
    name: str
    values: Data
    time_unit: str
    number_columns: list[str]
    subset_row_name: str
    subset_column_name: str

