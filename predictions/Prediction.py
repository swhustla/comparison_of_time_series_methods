from dataclasses import dataclass
from typing import TypeVar, Optional

Data = TypeVar("Data")

@dataclass
class PredictionData:
    method_name: str
    values: Data
    prediction_column_name: str
    ground_truth_values: Optional[Data]
    confidence_columns: Optional[list[str]]
    title: str
    plot_folder: str
    plot_file_name: str
    model_config: Optional[dict] = None
    number_of_iterations: Optional[int] = 1
    confidence_on_mean: Optional[bool] = True
    confidence_method: Optional[str] = "std"
    color: Optional[str] = "orange"