
from methods.predict import Predict
from data.Data import Dataset, Result
from data.load import Load
from data.report import Report
from data.india_pollution import india_pollution
from data.stock_prices import stock_prices
from data.list_of_tuples import list_of_tuples
from predictions.ARIMA import arima
from predictions.linear_regression import linear_regression
from measurements.get_metrics import get_metrics


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


def calculate_metrics(data: Dataset, prediction: Result):
    """
    Calculate the metrics for the given data and prediction.
    """
    return get_metrics(data.values, prediction.values)

def predict_and_measure(data: Dataset):
    output = __predictors[data.method](data.values)
    i = 0
    predictions = list()
    metrics = list()
    while i < data.count:
        # try:
        i += 1
        prediction = output
        print(prediction)
        predictions.append(prediction)
        metrics.append(calculate_metrics(data, prediction))
        # except:
        #     break
    return Report(data.method, data, predictions, metrics)


def generate_predictions(methods: list[str], datasets: list[str], count = 1):
    """
    Generate predictions for the given dataset using the given methods.
    """
    if count <= 0:
        raise ValueError("Count must be greater than 0.")
    for method in methods:
        for dataset in datasets:
            data = load_dataset(dataset)
            yield predict_and_measure(Dataset(data, method, count))




__datasets = [
    # "india_pollution", 
    # "stock_prices", 
    "list_of_tuples"
    ]

__methods = [
    "linear_regression", 
    # "arima"
    ]

# TODO: datasets is akin to board size

for report in generate_predictions(__methods, __datasets):
    print(report)
