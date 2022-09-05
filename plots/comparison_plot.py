import pandas as pd
from typing import Tuple, Optional

from matplotlib import pyplot as plt

from methods.plot import Figure, Plot
from data.Data import Dataset 
from predictions.Prediction import PredictionData

from methods.comparison_plot import comparison_plot as method


def __plot(
    ground_truth_series: pd.Series,
    prediction_series: pd.Series,
    confidence_interval_df: Optional[pd.DataFrame],
    title: str,
) -> Figure:
    """Plot the data, optionally with confidence intervals."""
    figure, ax = plt.subplots(figsize=(12, 6))

    ground_truth_series.plot(ax=ax, label="Ground truth")
    prediction_series.plot(ax=ax, label="Forecast")
    if confidence_interval_df is not None:
        ax.fill_between(
            x=prediction_series.index,
            y1=confidence_interval_df.iloc[:, 0],
            y2=confidence_interval_df.iloc[:, 1],
            alpha=0.2,
            color="orange",
            label="Confidence interval",
        )
    ax.set_title(title)

    ax.legend()

    return figure

def __save_plot(figure: Figure, title: str) -> None:
    """Save the plot to disk."""
    print(f"Saving plot for {title}")
    snake_case_title = title.replace(" ", "_").lower()
    figure.savefig(f"plots/{snake_case_title}.png")


comparison_plot = method(__plot, __save_plot)
