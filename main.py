
from methods.predict import Predict
from data.Data import Dataset
from data.report import Report
from predictions.ARIMA import arima
from predictions.linear_regression import linear_regression

__predictors: dict[str, Predict] = {
    "linear_regression": linear_regression,
    "ARIMA": arima,
}


def predict(data: Dataset):
    generator = __predictors[data.method](data.values)
    predictions = list()
    while i < data.count:
        try:
            i += 1
            prediction = generator.next()
            predictions.append(prediction)
        except:
            break
    return Report(data.method, data, predictions)


def generate_predictions(methods: list[str], datasets: list[str], count = 1):
    """
    Generate predictions for the given dataset using the given methods.
    """
    if count <= 0:
        raise ValueError("Count must be greater than 0.")
    for method in methods:
        for dataset in datasets:
            yield predict(Dataset(__datasets[dataset], method, count))


__datasets = ["india_pollution", "stock_prices", "list_of_tuples"]

__methods = ["linear_regression", "arima"]

# TODO: datasets is akin to board size

for report in generate_predictions(__methods, __datasets):
    print(report)
