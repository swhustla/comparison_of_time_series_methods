from typing import TypeVar
from methods.FCNN import fcnn as method
import pandas as pd

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, Dropout

from sklearn.model_selection import train_test_split

from data.Data import Dataset
from predictions.Prediction import PredictionData

Model = TypeVar("Model")

__dropout_ratio: float = 0.2
__hidden_layer_size: int = 100
__test_size: float = 0.2
__number_of_epochs: int = 1000

__callback = keras.callbacks.EarlyStopping(monitor="loss", patience=3)


def __create_model(data: Dataset) -> Model:
    """
    Create a model for the given data.
    """
    input_layer = Input(len(data.values.columns))

    hidden_layer = Dropout(__dropout_ratio)(input_layer)
    hidden_layer = Dense(__hidden_layer_size, activation="relu")(hidden_layer)

    output_layer = Dropout(__dropout_ratio)(hidden_layer)
    output_layer = Dense(1)(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        loss="mse",
        optimizer=keras.optimizers.Adam(),
        metrics=[
            keras.metrics.RootMeanSquaredError(),
            keras.metrics.MeanAbsoluteError(),
        ],
    )

    return model


def __split_data(
    data: Dataset,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the given data into training and testing data.
    """
    shift_amount = int(__test_size * len(data.values))

    x_train, x_test_val, y_train, y_test_val = train_test_split(
        data.values,
        data.values[data.subset_column_name].shift(-1 * shift_amount),
        test_size=__test_size * 2,
        shuffle=False,
    )

    x_test = x_test_val[:int(len(x_test_val) // 2)]
    x_val = x_test_val[int(len(x_test_val) // 2):]
    y_test = y_test_val[:int(len(y_test_val) // 2)]
    y_val = y_test_val[int(len(y_test_val) // 2):]

    print(f"\n\ny_val: {y_val}")

    return x_train, x_test, y_train, y_test, x_val, y_val


def __learn_model(model: Model, x_train: pd.DataFrame, y_train: pd.DataFrame) -> Model:
    """
    Train the given model on the given data.
    """
    print(f"Training model on {x_train.shape} and {y_train.shape} data")
    model.fit(
        x_train,
        y_train,
        epochs=__number_of_epochs,
        callbacks=[__callback],
        verbose=0,
    )

    return model


def __get_predictions(
    model: Model, data: Dataset, x_test: pd.DataFrame, y_val: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict using the given data with the given model.
    """
    title = f"{data.subset_column_name} forecast for {data.subset_row_name} with FCNN"
    prediction = model.predict(x_test)
    print(f"prediction_values: {prediction}, y_val: {y_val}")
    prediction_series = pd.Series(
        prediction.reshape(-1), index=y_val.index
    )
    return PredictionData(
        values=prediction_series,
        prediction_column_name=None,
        ground_truth_values=y_val,
        confidence_columns=None,
        title=title,
    )


fcnn = method(__create_model, __split_data, __learn_model, __get_predictions)
