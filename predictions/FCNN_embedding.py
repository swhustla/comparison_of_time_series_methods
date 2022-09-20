from typing import TypeVar
from methods.FCNN_embedding import fcnn_embedding as method
import pandas as pd

import tensorflow as tf
from keras.optimizers import Adam
from keras.metrics import MeanAbsoluteError, RootMeanSquaredError
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, Concatenate

from sklearn.model_selection import train_test_split

from data.Data import Dataset
from predictions.Prediction import PredictionData

Model = TypeVar("Model")


__test_size: float = 0.2
__number_of_epochs: int = 1000

__callback = keras.callbacks.EarlyStopping(monitor="loss", patience=3)


def __create_model(training_data: pd.DataFrame, data: Dataset) -> Model:
    """
    Create a model for the given data.
    With the use of an embedding layer, the model can handle categorical data.
    """
    scale = tf.constant(training_data[data.subset_column_name].std(), dtype=tf.float32)
    continuous_layer = Input(shape=1)

    categorical_months_layer = Input(shape=1)
    embedded_months_layer = Embedding(12, 5)(categorical_months_layer)
    embedded_flat_months_layer = Flatten()(embedded_months_layer)

    year_input = Input(shape=1)
    year_layer = Dense(1)(year_input)

    hidden_output = Concatenate(-1)(
        [embedded_flat_months_layer, year_layer, continuous_layer]
    )
    output_layer = Dense(1)(hidden_output)
    output = output_layer * scale + continuous_layer

    model = keras.Model(
        inputs=[continuous_layer, categorical_months_layer, year_input],
        outputs=output,
    )
    model.compile(
        loss="mse",
        optimizer=Adam(),
        metrics=[RootMeanSquaredError(), MeanAbsoluteError()],
    )

    return model


def __add_features(data: Dataset) -> Dataset:
    """
    Add features to the given data.
    """
    data.values = pd.DataFrame(
        data.values[data.subset_column_name],
        columns=[data.subset_column_name],
        index=data.values.index,
    )

    data.values["year"] = data.values.index.isocalendar().year.astype(
        int
    ) - data.values.index.isocalendar().year.min().astype(int)
    data.values["month"] = data.values.index.month.astype(int) - 1


    return data


def __split_data(
    data: Dataset,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the given data into training and testing data.
    """
    if data.time_unit == "days":
        shift_amount = 30
    elif data.time_unit == "weeks":
        shift_amount = 4
    elif data.time_unit == "months":
        shift_amount = 1
    else:
        raise ValueError(f"Unknown time unit {data.time_unit}")

    x_train, x_test, y_train, y_test = train_test_split(
        data.values,
        data.values[data.subset_column_name].shift(-shift_amount),
        test_size=__test_size,
        shuffle=False,
    )

    return x_train, x_test, y_train, y_test


def __learn_model(model: Model, x_train: pd.DataFrame, y_train: pd.DataFrame) -> Model:
    """
    Train the given model on the given data.
    """
    print(f"Training model on {x_train.shape} and {y_train.shape} data")
    model.fit(
        (x_train.iloc[:, 0], x_train["month"], x_train["year"]),
        y_train,
        epochs=__number_of_epochs,
        callbacks=[__callback],
        verbose=0,
    )

    return model


def __get_predictions(
    model: Model, data: Dataset, x_test: pd.DataFrame, y_test: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict using the given data with the given model.
    """
    title = f"{data.subset_column_name} forecast for {data.subset_row_name} with FCNN embedding"
    prediction = model.predict((x_test[data.subset_column_name], x_test["month"], x_test["year"]), verbose=0)
    prediction_series = pd.Series(prediction.reshape(-1), index=y_test.index)

    print(f"y_test sample: {y_test.head()}")
    return PredictionData(
        values=prediction_series,
        prediction_column_name=None,
        ground_truth_values=y_test,
        confidence_columns=None,
        title=title,
    )


fcnn_embedding = method(__add_features, __split_data, __create_model, __learn_model, __get_predictions)
