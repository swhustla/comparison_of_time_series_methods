
from methods.predict import Predict
from data.Data import Dataset, Result
from predictions.Prediction import PredictionData
from data.load import Load
from data.report import Report
from data.india_pollution import india_pollution
from data.stock_prices import stock_prices
from data.list_of_tuples import list_of_tuples
from predictions.ARIMA import arima
from predictions.linear_regression import linear_regression
from measurements.get_metrics import get_metrics
from plots.comparison_plot import comparison_plot


__dataset_loaders: dict[str, Load[Dataset]] = {
    "india_pollution": india_pollution,
    "stock_prices": stock_prices,
    "list_of_tuples": list_of_tuples, 
}



__predictors: dict[str, Predict[Dataset, Result]] = {
    "linear_regression": linear_regression,
    "ARIMA": arima,
}


def load_dataset(dataset_name: str):
    """
    Load the given dataset.
    """
    return __dataset_loaders[dataset_name]()


def calculate_metrics(data: Dataset, prediction: PredictionData):
    """
    Calculate the metrics for the given data and prediction.
    """
    return get_metrics(data.values, prediction.values)

def predict_measure_plot(data: Dataset, method_name: str, plot: bool = True) -> Report:
    """Generate a report for the given data and method."""

    prediction, ground_truth = __predictors[method_name](data)

    metrics = calculate_metrics(ground_truth, prediction)

    if plot:
        comparison_plot(ground_truth.values, prediction.values, prediction.confidence_columns, prediction.title)
 
    return Report(method_name, data, prediction, metrics)


def generate_predictions(methods: list[str], datasets: list[str]):
    """
    Generate predictions for the given dataset using the given methods.
    """
    for method_name in methods:
        for dataset_name in datasets:
            data, time_unit, number_columns, subset_row_name, subset_column_name = load_dataset(dataset_name)
            yield predict_measure_plot(Dataset(data, time_unit, number_columns, subset_row_name, subset_column_name), method_name)




__datasets = [
    "india_pollution", 
    "stock_prices", 
    "list_of_tuples"
    ]

__methods = [
    "linear_regression", 
    # "arima"
    ]

# TODO: datasets is akin to board size

for report in generate_predictions(__methods, __datasets):
    print(report)
