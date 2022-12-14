import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from data.dataset import Dataset

from matplotlib import pyplot as plt

from methods.plot import Figure
from predictions.Prediction import PredictionData

from methods.comparison_plot_multi import comparison_plot_multi as method


def __get_prediction_series(prediction: PredictionData) -> pd.Series:
    if type(prediction.values) is pd.DataFrame:
        if prediction.prediction_column_name is not None:
            prediction_series = prediction.values[prediction.prediction_column_name]
        else:
            prediction_series = prediction.values.iloc[:, 0]
    else:
        prediction_series = prediction.values
    return prediction_series


def __get_plot_metadata(
    prediction_data: PredictionData,
) -> Tuple[str, str, str]:
    """Get the metadata for the multi plot."""
    prediction_data = prediction_data[0]
    method_specific_plot_folder = prediction_data.plot_folder
    # get rid of the last folder
    generic_plot_folder = "/".join(method_specific_plot_folder.split("/")[:-2])
    plot_folder = f"{generic_plot_folder}/comparison_plot_multi/"
    method_specific_file_name = prediction_data.plot_file_name
    # use the first two words of the file name
    generic_file_name = "_".join(method_specific_file_name.split("_")[:2])
    # add a timestamp to the file name
    current_time_string = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    plot_file_name = f"{generic_file_name}_comparison_plot_multi_{current_time_string}"
    if "with" in prediction_data.title:
        title = (prediction_data.title.split("with")[0]).strip()
    elif "using" in prediction_data.title:
        title = prediction_data.title.split("using")[0].strip()
    else:
        title = prediction_data.title

    return plot_folder, plot_file_name, title


def __plot_full_dataset_plus_predictions(
    training_data: pd.DataFrame, predictions: list[PredictionData]
) -> Figure:
    """Plot the full data and the prediction."""
    figure, axis = plt.subplots(figsize=(10, 5))
  
    training_data_series = training_data.iloc[:, 0]
    training_data_series.plot(ax=axis, label="Training data", style=".", c="blue")
    

    ground_truth_values = predictions[0].ground_truth_values
    ground_truth_values.plot(
        ax=axis, label="Ground truth", style="x", c="blue", alpha=0.5
    )
    
    for prediction in predictions:
        prediction_series = __get_prediction_series(prediction)
        if type(prediction_series) is np.ndarray:
            prediction_series = pd.Series(
                prediction_series, index=prediction.ground_truth_values.index
            )
        prediction_series.plot(
            ax=axis,
            #label=prediction.method_name + " out-of-sample forecast",
            label=prediction.method_name,
            style="-",
            c=prediction.color,
        )
        if prediction.in_sample_prediction is not None:
            prediction.in_sample_prediction.plot(
                ax=axis,
                #label=prediction.method_name + " in-sample prediction",
                label="_nolegend_",
                style="-",
                c=prediction.color,
            )

    axis.legend(loc="upper left")

    return figure


def __save_plot(
    figure: Figure, folder: str, file_name: str, plot_type: str, title: str
) -> None:
    """Save the plot to disk."""
    print(f"Saving plot to {folder}/{file_name}_{plot_type}.png")
    if not os.path.exists(f"plots/{folder}"):
        os.makedirs(f"plots/{folder}")

    figure.savefig(f"plots/{folder}{file_name}_{plot_type}.png", bbox_inches="tight")


comparison_plot_multi = method(
    __plot_full_dataset_plus_predictions, __get_plot_metadata, __save_plot
)
