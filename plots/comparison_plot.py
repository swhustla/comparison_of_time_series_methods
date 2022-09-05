import pandas as pd
from typing import Tuple, Optional

from matplotlib import pyplot as plt

from methods.comparison_plot import comparison_plot as method


def __plot(
    ground_truth_series: pd.Series,
    prediction_series: pd.Series,
    confidence_interval_df: Optional[pd.DataFrame],
    title: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Plot the data, optionally with confidence intervals."""
    _, ax = plt.subplots(figsize=(12, 6))

    ground_truth_series.plot(ax=ax, label="Ground truth")
    prediction_series.plot(ax=ax, label="Forecast")
    if confidence_interval_df:
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


comparison_plot = method(__plot)
