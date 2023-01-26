from dataclasses import dataclass
from data.dataset import Dataset
from predictions.Prediction import PredictionData


@dataclass(frozen=True)
class Report:
    tstart: float
    method: str
    dataset: Dataset
    prediction: PredictionData
    metrics: dict[str, float]