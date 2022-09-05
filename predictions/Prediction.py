from dataclasses import dataclass
from typing import TypeVar, Optional

Data = TypeVar("Data")

@dataclass
class PredictionData:
    values: Data
    confidence_columns: Optional[list[str]]
    title: str