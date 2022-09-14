from dataclasses import dataclass
from typing import TypeVar, Optional

Data = TypeVar("Data")

@dataclass
class PredictionData:
    values: Data
    prediction_column_name: str
    ground_truth_values: Optional[Data]
    confidence_columns: Optional[list[str]]
    title: str