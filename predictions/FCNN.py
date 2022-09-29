""" 
Fully connected neural network (FCNN) prediction model.

This model uses a fully connected neural network to predict the future values
 of a given time series. 
A fully connected neural network is a neural network where each node in one
    layer is connected to each node in the next layer.

Its layers are:
    - Input layer (size of the number of features)
    - Dropout layer (to prevent overfitting)
    - Dense layer (with relu activation function)
    - Dropout layer (to prevent overfitting)
    - Dense layer (with linear activation function)

The model is trained on the given data using the Adam optimizer and two loss
    functions:
    - Mean squared error (MSE)
    - Mean absolute error (MAE)

The model is trained for a fixed number of epochs (see __number_of_epochs).

Model training involves a validation step, where the model is evaluated on a
    validation set. If the validation loss does not improve for a fixed number
    of epochs (see __patience), the training is stopped early.

The data is split into training and testing data using a fixed test size and
    without shuffling (see __test_size).
A fixed test size is used to ensure that the same data is used for testing.


"""


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

def __add_features(data: Dataset) -> Dataset:
    """
    Add features to the given data.
    """
    data.values = pd.DataFrame(data.values[data.subset_column_name], columns=[data.subset_column_name], index=data.values.index)

    
    data.values["year"] = data.values.index.isocalendar().year.astype(int)
    data.values["month"] = data.values.index.month.astype(int)


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
        data.values.shift(-1 * shift_amount),
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
        x_train,
        y_train,
        epochs=__number_of_epochs,
        callbacks=[__callback],
        verbose=0,
    )

    return model


def __get_predictions(
    model: Model, data: Dataset, x_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
    """
    Predict using the given data with the given model.
    """
    title = f"{data.subset_column_name} forecast for {data.subset_row_name} with FCNN"
    prediction = model.predict(x_test)
    prediction_series = pd.Series(
        prediction.reshape(-1), index=y_test.index
    )

    return PredictionData(
        values=prediction_series,
        prediction_column_name=None,
        ground_truth_values=y_test[data.subset_column_name],
        confidence_columns=None,
        title=title,
        plot_folder=f"{data.name}/{data.subset_row_name}/FCNN/",
        plot_file_name=f"{data.subset_column_name}_forecast",

    )


fcnn = method(__create_model, __add_features, __split_data, __learn_model, __get_predictions)
