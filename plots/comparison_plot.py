import os
import logging

import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Union
from pandas import Timestamp

from matplotlib import pyplot as plt
from numpy import nan
import pandas as pd

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
) -> Tuple[pd.Series, pd.Series, str]:
    if prediction.confidence_columns is None:
        logging.info(
            f"\n\nPlotting {prediction.title} without pre-defined confidence intervals\n\n"
        )
        upper_limit = prediction_series + prediction.values.std()
        lower_limit = prediction_series - prediction.values.std()
        confidence_interval_label = "Confidence interval - 1 S.D."
    else:
        if prediction.confidence_on_mean == False:
            logging.info(
                f"\n\nPlotting {prediction.title} with pre-defined confidence intervals\n\n"
            )
            upper_limit = prediction.values.loc[:, prediction.confidence_columns[1]]
            lower_limit = prediction.values.loc[:, prediction.confidence_columns[0]]
            confidence_interval_label = prediction.confidence_method
        elif prediction.confidence_on_mean == True:
            logging.info(
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


__who_recommendation = 5
__india_recommendation = 40


def __add_india_who_recommendation(
    axis: plt.Axes,
) -> Tuple[plt.Axes, plt.Line2D, plt.Line2D]:
    """Add India and WHO recommendation to the plot."""
    line_one = axis.axhline(
        y=__india_recommendation, color="r", linestyle="--", linewidth=2
    )
    line_two = axis.axhline(
        y=__who_recommendation, color="darksalmon", linestyle="--", linewidth=2
    )
    twenty_percent_along_x_axis = (axis.get_xlim()[1] - axis.get_xlim()[0]) * 0.2
    time_stamp_for_india_recommendation = (
        axis.get_xlim()[0] + twenty_percent_along_x_axis
    )
    five_percent_along_y_axis = (axis.get_ylim()[1] - axis.get_ylim()[0]) * 0.05
    location_who = __who_recommendation + five_percent_along_y_axis
    location_india = __india_recommendation + five_percent_along_y_axis
    axis.text(
        time_stamp_for_india_recommendation,
        location_india,
        "India",
        color="r",
        ha="right",
        va="center",
    )
    axis.text(
        time_stamp_for_india_recommendation,
        location_who,
        "WHO",
        color="darksalmon",
        ha="right",
        va="center",
    )

    return axis, line_one, line_two


def __convert_string_to_series(
    data_string: str,
) -> Union[pd.Series, pd.DataFrame, List]:
    """Convert ground truth, prediction values, in sample prediction to correct format from string"""

    try:
        tidy_series = pd.Series(eval(data_string))
    except SyntaxError:
        try:
            logging.debug("Series conversion failed; trying to convert to DataFrame")
            tidy_series = pd.DataFrame(eval(data_string), index=[0])
        except SyntaxError:
            logging.debug("DataFrame conversion failed; trying to convert to list")
            tidy_series = list(eval(data_string))
        except Exception as e:
            logging.debug(f"Failed to convert to list: {e}")
    except Exception as e:
        logging.debug(f"Failed to convert to series: {e}")

    return tidy_series


def __tidy_up_prediction_data_object(
    predictions: PredictionData,
) -> PredictionData:
    # pred_data = []
    # for predictions in predictions:
    cleaned_pred_data = PredictionData(
        method_name=predictions.method_name,
        values=predictions.values,
        prediction_column_name=predictions.prediction_column_name,
        ground_truth_values=predictions.ground_truth_values,
        confidence_columns=predictions.confidence_columns,
        title=predictions.title,
        plot_folder=predictions.plot_folder,
        plot_file_name=predictions.plot_file_name,
        model_config=predictions.model_config,
        number_of_iterations=predictions.number_of_iterations,
        confidence_on_mean=predictions.confidence_on_mean,
        confidence_method=predictions.confidence_method,
        color=predictions.color,
        in_sample_prediction=predictions.in_sample_prediction,
    )

    cleaned_pred_data.ground_truth_values = __convert_string_to_series(
        predictions.ground_truth_values
    )

    try:
        cleaned_pred_data.values = pd.DataFrame(eval(predictions.values))
    except:
        cleaned_pred_data.values = __convert_string_to_series(predictions.values)
    try:
        cleaned_pred_data.in_sample_prediction = __convert_string_to_series(
            predictions.in_sample_prediction
        )
    except:
        cleaned_pred_data.in_sample_prediction = None
            

    return cleaned_pred_data


def __full_data_plus_prediction_plot(
    training_data: pd.DataFrame, prediction: PredictionData
) -> Figure:
    """Plot the full data and the prediction."""

    figure, axis = plt.subplots(figsize=(12, 7))

    logging.debug(f"type of training data: {type(training_data)}")
    logging.debug(f"size of training data: {training_data.shape}")
    logging.debug(f"samples of training data: {training_data[:5]}")

    if type(training_data) is pd.DataFrame:
        training_data_series = training_data.iloc[:, 0]

    training_data_series.plot(ax=axis, label="Training data", style=".", c="blue")

    try:
        prediction = __tidy_up_prediction_data_object(predictions=prediction)
    except:
        ground_truth_values = prediction.ground_truth_values

    ground_truth_values = prediction.ground_truth_values
    title = prediction.title + ' ' + prediction.plot_folder.split("/")[1]

    ground_truth_values.plot(
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
    upper_limit.plot(color=prediction.color,alpha=0.2,label="_nolegend_",)
    lower_limit.plot(color=prediction.color,alpha=0.2,label="_nolegend_",)

    # make index compatible with matplotlib
    dates_for_index = prediction_series.index.values

    axis.fill_between(
        x=dates_for_index,
        y1=lower_limit,
        y2=upper_limit,
        alpha=0.2,
        color=prediction.color,
        label=confidence_interval_label,
    )


    if training_data.columns[0] == "PM2.5":
        axis, line_1, line_2 = __add_india_who_recommendation(axis)
        # Create a legend
        second_legend = plt.legend(
            handles=[line_1, line_2],
            labels=[r"India$^*$", r"WHO$^\dagger$"],
            loc=1,
            ncol=2,
            title="Recommendation:",
        )
        # Add the legend manually to the current Axes.
        plt.gca().add_artist(second_legend)
        axis.annotate(
            r"* = Indian National Ambient Air Quality Standards, annual average PM2.5 threshold 40[$\mu g/m^3$]"
            + "\n"
            + r"$\dagger$ = World Health Organization, annual average PM2.5 threshold 5[$\mu g/m^3$]",
            xy=(0.0, 0),
            xytext=(0, 0),
            xycoords=("axes fraction", "figure fraction"),
            textcoords="offset points",
            size=8,
            ha="left",
            va="bottom",
            annotation_clip=False,
        )

    axis.set_title(title)
    axis.set_xlabel("Date")
    if training_data.columns[0] == "PM2.5":
        axis.set_ylabel(f"{training_data.columns[0]} [$\mu g/m^3$]")
    else:
        axis.set_ylabel(f"{training_data.columns[0]}")

    axis.set_ylim(
        bottom=0,
        top=1.1
        * max(
            training_data_series.max(),
            prediction_series.max(),
            ground_truth_values.max(),
        ),
    )
    axis.legend(loc="upper left")

    return figure


def __plot(
    training_data: pd.DataFrame, prediction: PredictionData
) -> Figure:
    """Plot the full data and the prediction."""

    figure, axis = plt.subplots(figsize=(12, 7))

    logging.debug(f"type of training data: {type(training_data)}")
    logging.debug(f"size of training data: {training_data.shape}")
    logging.debug(f"samples of training data: {training_data[:5]}")

    if type(training_data) is pd.DataFrame:
        training_data_series = training_data.iloc[:, 0]

    try:
        prediction = __tidy_up_prediction_data_object(predictions=prediction)
    except:
        ground_truth_values = prediction.ground_truth_values

    ground_truth_values = prediction.ground_truth_values
    title = prediction.title + ' ' + prediction.plot_folder.split("/")[1]

    ground_truth_values.plot(
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

    (
        upper_limit,
        lower_limit,
        confidence_interval_label,
    ) = __figure_out_confidence_interval_plot(prediction, prediction_series)
    upper_limit.plot(color=prediction.color,alpha=0.2,label="_nolegend_")
    lower_limit.plot(color=prediction.color,alpha=0.2,label="_nolegend_")
    # make index compatible with matplotlib
    dates_for_index = prediction_series.index.values 
    axis.fill_between(
        x=dates_for_index,
        y1=lower_limit,
        y2=upper_limit,
        alpha=0.2,
        color=prediction.color,
        label=confidence_interval_label,
    )

    if training_data.columns[0] == "PM2.5":
        axis, line_1, line_2 = __add_india_who_recommendation(axis)
        # Create a legend
        second_legend = plt.legend(
            handles=[line_1, line_2],
            labels=[r"India$^*$", r"WHO$^\dagger$"],
            loc=1,
            ncol=2,
            title="Recommendation:",
        )
        # Add the legend manually to the current Axes.
        plt.gca().add_artist(second_legend)
        axis.annotate(
            r"* = Indian National Ambient Air Quality Standards, annual average PM2.5 threshold 40[$\mu g/m^3$]"
            + "\n"
            + r"$\dagger$ = World Health Organization, annual average PM2.5 threshold 5[$\mu g/m^3$]",
            xy=(0.0, 0),
            xytext=(0, 0),
            xycoords=("axes fraction", "figure fraction"),
            textcoords="offset points",
            size=8,
            ha="left",
            va="bottom",
            annotation_clip=False,
        )

    axis.set_title(title)
    axis.set_xlabel("Date")
    if training_data.columns[0] == "PM2.5":
        axis.set_ylabel(f"{training_data.columns[0]} [$\mu g/m^3$]")
    else:
        axis.set_ylabel(f"{training_data.columns[0]}")

    axis.set_ylim(
        bottom=0, top=1.1 * max(ground_truth_values.max(), prediction_series.max())
    )
    axis.legend(loc="upper left")

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
