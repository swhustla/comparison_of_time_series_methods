import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from matplotlib import pyplot as plt

from methods.plot import Figure
from predictions.Prediction import PredictionData

from methods.comparison_plot import comparison_plot as method

def __get_prediction_series(prediction: PredictionData) -> pd.Series:
    if type(prediction.values) is pd.DataFrame:
        if prediction.prediction_column_name is not None:
            prediction_series = prediction.values[prediction.prediction_column_name]
        else:
            prediction_series = prediction.values.iloc[:, 0]
    else:
        prediction_series = prediction.values
    return prediction_series

def __full_data_plus_prediction_plot(training_data: pd.DataFrame, prediction: PredictionData) -> Figure:
    """Plot the full data and the prediction."""
    title = prediction.title
    figure, axes = plt.subplots(figsize=(10, 5))

    training_data_series = training_data.iloc[:, 0]
    training_data_series.plot(ax=axes, label="Training data", style=".")

    prediction_series = __get_prediction_series(prediction)
    if type(prediction_series) is np.ndarray:
        prediction_series = pd.Series(prediction_series, index=prediction.ground_truth_values.index)
    prediction_series.plot(ax=axes, label="Forecast", style="-")

    if prediction.confidence_columns is not None:
        confidence_interval_df = prediction.values.loc[:, prediction.confidence_columns]
        axes.fill_between(
            x=prediction.values.index,
            y1=confidence_interval_df.iloc[:, 0],
            y2=confidence_interval_df.iloc[:, 1],
            alpha=0.2,
            color="orange",
            label="Confidence interval",
        )
    axes.set_title(title)

    axes.set_ylim(bottom=0, top=1.1 * max(training_data_series.max(), prediction_series.max()))
    axes.legend()
    return figure


def __plot(
    prediction: PredictionData,
) -> Figure:
    """Plot the data, optionally with confidence intervals."""
    ground_truth_series = prediction.ground_truth_values
    prediction_series = __get_prediction_series(prediction)

    if type(prediction_series) is np.ndarray:
        prediction_series = pd.Series(prediction_series, index=prediction.ground_truth_values.index)


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
    if type(ground_truth_series) is pd.DataFrame:
        ground_truth_series = ground_truth_series.iloc[:, 0]
    print(f"ground_truth_series.max() = {ground_truth_series.max()}")
    print(f"prediction_series.max() = {prediction_series.max()}")
    ax.set_ylim(bottom=0, top=1.1 * max(ground_truth_series.max(), prediction_series.max()))

    ax.legend()

    return figure


def __save_plot(figure: Figure, folder: str, file_name: str, plot_type: str, title: str) -> None:
    """Save the plot to disk."""
    print(f"Saving {plot_type} plot for {title} to {folder}/{file_name}.png")
    if not os.path.exists(f"plots/{folder}"):
        os.makedirs(f"plots/{folder}")
        
    figure.savefig(f"plots/{folder}/{file_name}_{plot_type}.png", bbox_inches="tight")


comparison_plot = method(__full_data_plus_prediction_plot, __plot, __save_plot)
