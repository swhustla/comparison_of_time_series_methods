"""Plot a comparison of the ground truth and forecast."""

from typing import Callable, TypeVar

from methods.linear_regression import Prediction

Data = TypeVar("Data", contravariant=True)
Prediction = TypeVar("Prediction")
Plot = TypeVar("Plot", covariant=True)

from .plot import Plot

def comparison_plot(
    plot: Callable[[Data, Prediction], Plot],
) -> Plot[Data, Prediction, Plot]:
    def plot_comparison(
        ground_truth: Data,
        prediction: Data,
    ) -> Plot:
        return plot(ground_truth, prediction)

    return plot_comparison