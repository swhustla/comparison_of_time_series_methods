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


def __figure_out_confidence_interval_plot(
    prediction: PredictionData, prediction_series: pd.Series
) -> Tuple[float, float, str]:
    if prediction.confidence_columns is None:
        print(
            f"\n\nPlotting {prediction.title} without pre-defined confidence intervals\n\n"
        )
        upper_limit = prediction_series + prediction.values.std()
        lower_limit = prediction_series - prediction.values.std()
        confidence_interval_label = "Confidence interval - 1 S.D."
    else:
        if prediction.confidence_on_mean == False:
            print(
                f"\n\nPlotting {prediction.title} with pre-defined confidence intervals\n\n"
            )
            upper_limit = prediction.values.loc[:, prediction.confidence_columns[1]]
            lower_limit = prediction.values.loc[:, prediction.confidence_columns[0]]
            confidence_interval_label = prediction.confidence_method
        elif prediction.confidence_on_mean == True:
            print(
                f"\n\nPlotting {prediction.title} with pre-defined confidence intervals\n\n"
            )
            upper_limit = prediction_series + (
                prediction.values["mean"]
                - prediction.values.loc[:, prediction.confidence_columns[0]]
            )
            lower_limit = prediction_series - (
                prediction.values.loc[:, prediction.confidence_columns[1]]
                - prediction.values["mean"]
            )
            confidence_interval_label = prediction.confidence_method
    return upper_limit, lower_limit, confidence_interval_label


def __full_data_plus_prediction_plot(
    training_data: pd.DataFrame, prediction: PredictionData
) -> Figure:
    """Plot the full data and the prediction."""
    title = prediction.title
    figure, axis = plt.subplots(figsize=(10, 5))

    training_data_series = training_data.iloc[:, 0]

    training_data_series.plot(ax=axis, label="Training data", style=".", c="blue")

    ground_truth_series = prediction.ground_truth_values
    ground_truth_series.plot(
        ax=axis, label="Ground truth", style="x", color="blue", alpha=0.5
    )

    prediction_series = __get_prediction_series(prediction)
    if type(prediction_series) is np.ndarray:
        if type(prediction.ground_truth_values) is pd.DataFrame:
            prediction_series = pd.Series(
                prediction_series, index=prediction.ground_truth_values.index
            )
        elif type(prediction.ground_truth_values) is np.array:
            prediction_series = pd.Series(
                prediction_series, index=training_data_series.index
            )

    try:
        prediction_series.plot(
            ax=axis,
            label=f"{prediction.method_name} prediction",
            style="-",
            c=prediction.color,
        )
    except Exception:
        # make index compatible with matplotlib
        prediction_series.index = pd.to_datetime(prediction_series.index)
        prediction_series.plot(
            ax=axis,
            label=f"{prediction.method_name} prediction",
            style="-",
            c=prediction.color,
        )

    if prediction.in_sample_prediction is not None:
        prediction.in_sample_prediction.plot(
            ax=axis,
            label="_nolegend_",
            style="-",
            c=prediction.color,
            alpha=0.7,
        )

    (
        upper_limit,
        lower_limit,
        confidence_interval_label,
    ) = __figure_out_confidence_interval_plot(prediction, prediction_series)
    axis.fill_between(
        x=prediction.values.index,
        y1=lower_limit,
        y2=upper_limit,
        alpha=0.2,
        color="orange",
        label=confidence_interval_label,
    )
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel(f"{training_data.columns[0].capitalize()}")

    axis.set_ylim(
        bottom=0, top=1.1 * max(training_data_series.max(), prediction_series.max())
    )
    axis.legend()
    return figure


def __plot(
    prediction: PredictionData,
) -> Figure:
    """Plot the data, optionally with confidence intervals."""
    ground_truth_series = prediction.ground_truth_values
    prediction_series = __get_prediction_series(prediction)

    if type(prediction_series) is np.ndarray:
        prediction_series = pd.Series(
            prediction_series, index=prediction.ground_truth_values.index
        )

    (
        upper_limit,
        lower_limit,
        confidence_interval_label,
    ) = __figure_out_confidence_interval_plot(prediction, prediction_series)

    figure, ax = plt.subplots(figsize=(12, 6))

    ground_truth_series.plot(ax=ax, label="Ground truth")
    prediction_series.plot(ax=ax, label=f"{prediction.method_name} forecast", c=prediction.color)

    # TODO: add confidence interval around the prediction, not the mean

    ax.fill_between(
        x=prediction_series.index,
        y1=lower_limit,
        y2=upper_limit,
        alpha=0.2,
        color=prediction.color,
        label=confidence_interval_label,
    )
    ax.set_title(prediction.title)
    if type(ground_truth_series) is pd.DataFrame:
        ground_truth_series = ground_truth_series.iloc[:, 0]

    ax.set_xlabel("Date")
    ax.set_ylabel(f"{prediction_series.columns[0].capitalize()}")

    ax.set_ylim(
        bottom=0, top=1.1 * max(ground_truth_series.max(), prediction_series.max())
    )

    ax.legend()

    return figure


def __save_plot(
    figure: Figure, folder: str, file_name: str, plot_type: str, title: str
) -> None:
    """Save the plot to disk."""
    print(f"Saving {plot_type} plot for {title} to {folder}{file_name}.png")
    if not os.path.exists(f"plots/{folder}"):
        os.makedirs(f"plots/{folder}")

    figure.savefig(f"plots/{folder}{file_name}_{plot_type}.png", bbox_inches="tight")


comparison_plot = method(__full_data_plus_prediction_plot, __plot, __save_plot)
