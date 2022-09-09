import pandas as pd
from typing import Tuple, Optional

from matplotlib import pyplot as plt

from methods.plot import Figure
from predictions.Prediction import PredictionData

from methods.comparison_plot import comparison_plot as method


def __plot(
    prediction: PredictionData,
) -> Figure:
    """Plot the data, optionally with confidence intervals."""
    ground_truth_series = prediction.ground_truth_values
    if type(prediction.values) is pd.DataFrame:
        if prediction.prediction_column_name is not None:
            prediction_series = prediction.values[prediction.prediction_column_name]
        else:
            prediction_series = prediction.values.iloc[:, 0]
    else:
        prediction_series = prediction.values

    title = prediction.title

    figure, ax = plt.subplots(figsize=(12, 6))

    ground_truth_series.plot(ax=ax, label="Ground truth")
    prediction_series.plot(ax=ax, label="Forecast")
    if prediction.confidence_columns is not None:
        confidence_interval_df = prediction.values.loc[:, prediction.confidence_columns]
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
