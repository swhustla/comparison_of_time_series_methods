import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional

from data.dataset import Dataset

from matplotlib import pyplot as plt

from methods.plot import Figure
from predictions.Prediction import PredictionData

from methods.comparison_plot_multi import comparison_plot_multi as method
from plots.comparison_plot import __add_india_who_recommendation

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
    training_data: pd.DataFrame, predictions: list[PredictionData], title: str
) -> Figure:
    """Plot the full data and the prediction."""
    figure, axis = plt.subplots(figsize=(12, 7))

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

    if training_data.columns[0] == "PM2.5":
        axis, line_1, line_2 = __add_india_who_recommendation(axis)     
        # Create a legend 
        second_legend = plt.legend(handles=[line_1,line_2],labels=[r"India$^*$",r"WHO$^\dagger$"],loc=1,ncol=2,title='Recommendation:')
        # Add the legend manually to the current Axes.
        plt.gca().add_artist(second_legend)
        axis.annotate(r"* = Indian National Ambient Air Quality Standards, annual average PM2.5 threshold 40[$\mu g/m^3$]"+"\n"+
                        r"$\dagger$ = World Health Organization, annual average PM2.5 threshold 5[$\mu g/m^3$]",
            xy=(0., 0), xytext=(0, 0),
            xycoords=('axes fraction', 'figure fraction'),
            textcoords='offset points',
            size=8, ha='left', va='bottom',
            annotation_clip=False)


    axis.legend(loc="upper left")
    axis.set_xlabel("Date")
    if training_data.columns[0] == "PM2.5":
        axis.set_ylabel(f"{training_data.columns[0]} [$\mu g/m^3$]")
    else:
        axis.set_ylabel(f"{training_data.columns[0]}")

    axis.set_title(title)
    axis.set_ylim(
        bottom=0, top=1.1 * max(training_data_series.max(), prediction_series.max())
    )

   


    return figure


def __save_plot(figure: Figure, folder: str, file_name: str, plot_type: str) -> None:
    """Save the plot to disk."""
    print(f"Saving plot to {folder}/{file_name}_{plot_type}.png")
    if not os.path.exists(f"plots/{folder}"):
        os.makedirs(f"plots/{folder}")

    figure.savefig(f"plots/{folder}{file_name}_{plot_type}.png", bbox_inches="tight")


comparison_plot_multi = method(
    __get_plot_metadata, __plot_full_dataset_plus_predictions, __save_plot
)
