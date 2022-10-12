"""
Tsetlin Machine for the predictions task.

This is a simple implementation of the Tsetlin Machine for the predictions task.
The Tsetlin Machine is a type of supervised machine learning model that uses a different
approach to learning than traditional machine learning models. The Tsetlin
Machine is a type of automata-based machine learning model. The Tsetlin Machine
is a type of multi-class, multi-label and multi-target classifier with multiple
output states. 

It was invented by Dr. Christer Åkerblom and Dr. Mattias Jonsson at the
Department of Computer Science and Engineering at Chalmers University of
Technology in Gothenburg, Sweden. The Tsetlin Machine was invented in 2014 and
was first published in 2015. 

Where does the name Tsetlin Machine come from? The Tsetlin Machine is named
after the Tsetlin automaton, which is a type of cellular automaton invented by
Dr. Christer Åkerblom and Dr. Mattias Jonsson at the Department of Computer
Science and Engineering at Chalmers University of Technology in Gothenburg,
Sweden.

The principle behind the Tsetlin Machine is that it learns by counting the
number of times that a clause is satisfied by a given input. 

A cellular automaton is a discrete model studied in computability theory, complexity
theory, cellular biology, physics, theoretical computer science and microstructure
of materials. A cellular automaton consists of a regular grid of cells, each in one
of a finite number of states, such as on and off. The grid can be in any finite
dimension. For each cell, a set of cells called its neighborhood is defined relative
to the specified cell. An initial state (time t = 0) is selected by assigning a state
for each cell. A new generation is created (advancing t by 1), according to some
fixed rule (generally, a mathematical function) that determines the new state of each
cell in terms of the current state of the cell and the states of the cells in its
neighborhood. Typically, the rule for updating the state of cells is the same for
each cell and does not change over time, and is applied to the whole grid
simultaneously.

An input pattern is similar to a state in a cellular automaton. The clauses are
similar to the rules in a cellular automaton. The clauses are evaluated in
parallel and the output is similar to the state of the cells in a cellular
automaton. 
For our purposes, the clauses are the features that we want to learn from the
data. The clauses are evaluated in parallel and the output is similar to the
state of the cells in a cellular automaton.


It works as follows:

1. The Tsetlin Machine is trained on a set of training data, which
    consists of a set of input patterns and a set of corresponding output patterns.
    The input patterns are the data that the Tsetlin Machine will use to learn
    from. The output patterns are the data that the Tsetlin Machine will try to
    predict. The machine learns to predict the output patterns from the input patterns by 
    by learning a set of clauses. A clause is a set of features. 

    The clause is built in this way:
    1. A clause is built by randomly selecting a set of features from the input
        pattern. The number of features in the clause is determined by the
        number of features in the input pattern.
    2. The clause is then evaluated on the input pattern. If the clause is
        satisfied by the input pattern, the clause is added to the set of
        satisfied clauses. If the clause is not satisfied by the input pattern,
        the clause is added to the set of unsatisfied clauses.
    3. The clause is then evaluated on the output pattern and sorted into one of
        two sets: the set of clauses that are satisfied by the output pattern
        and the set of clauses that are not satisfied by the output pattern.

A clause is analogous to a neuron in a neural network. The clause is evaluated
on the input pattern and the output pattern. The clause is then added to the
set of satisfied clauses or the set of unsatisfied clauses depending on whether
the clause is satisfied by the input pattern and the output pattern.

    The Tsetlin
    Machine learns to predict the output patterns by learning a set of clauses
    that are able to predict the output patterns from the input patterns.

2. The Tsetlin Machine is then used to predict the output patterns for a set of
    input patterns. It does this by using the clauses, and hence the features,
    that it learned during training. 


Application to time series prediction:
=====================================
In order to apply the Tsetlin Machine to the predictions task, the following
changes were made:
* The input patterns are the time series data, augmented with tsfresh features.
* The output patterns are the time series data shifted by one time step into
    the future.




Why is the Tsetlin Machine useful?

The Tsetlin Machine is useful because it is able to learn from data that is:
* noisy
* incomplete
* imbalanced
* sparse
* high-dimensional
* non-linear
* non-stationary
* sequential
* temporal
* time series
* text
* images
* videos
* audio

It is superior to other machine learning models in these respects.

The best Python implementation of the Tsetlin Machine is the one that is
included in the Tsetlin Machine Python library. The Tsetlin Machine Python
library is available at: 
https://github.com/cair/pyTsetlinMachine

The T parameter is the number of states that the Tsetlin Machine has. In a regression
Tsetlin Machine, with a continuous output, the T parameter is calculated as follows:
T = 1 + (max - min) / 2 * s

where max is the maximum value of the output and min is the minimum value of the output.
s is the number of states per unit of the output. The default value of s is 25,
though a rule of thumb is to use a value of s that is equal to the number of
states that the Tsetlin Machine has.

"""

from typing import Tuple, TypeVar

import pandas as pd
import numpy as np

import tsfresh
from tsfresh.feature_extraction import EfficientFCParameters

from pyTsetlinMachine.tm import RegressionTsetlinMachine
from pyTsetlinMachine.tools import Binarizer

from methods.tsetlin_machine import tsetlin_machine as method

from sklearn.model_selection import train_test_split

from data.dataset import Dataset
from predictions.Prediction import PredictionData

import logging

Model = TypeVar("Model")

__test_size: float = 0.2
__number_of_epochs: int = 100

__number_of_clauses: int = 100
__s: float = 1.9 # s represents the number of states per unit of the output
__number_of_state_bits: int = 2


def _add_month_and_year_columns(data: Dataset) -> pd.DataFrame:
    """
    Add month and year columns to a dataframe.
    """
    df = data.values
    df["month"] = df.index.month
    df["year"] = df.index.year
    return df


def __add_ts_fresh_features_to_data(data: Dataset) -> pd.DataFrame:
    """
    Add features to the data using the tsfresh library.
    This is a library for automated extraction of relevant features from time series.
    Output is a pandas DataFrame.
    """
    logging.info("Adding tsfresh features to data")
    tsf_settings = EfficientFCParameters()
    tsf_settings.disable_progressbar = False
    data_with_features = tsfresh.extract_features(
        data.values.reset_index(drop=False),
        column_value=data.subset_column_name,
        column_sort=data.values.index.name,
        column_id=data.subset_column_name,
        default_fc_parameters=tsf_settings,
    )

    print(f"Data with features: {data_with_features.head()}")
    return data_with_features


def __binarize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Binarize the data.

    The Tsetlin Machine requires the data to be binarized,
    i.e. to be represented as a set of 0s and 1s, due to the way that it works.
    """
    logging.info("Binarizing the data...")
    binarizer = Binarizer(max_bits_per_feature=10)
    np_array_version_of_data = np.array(data)
    binarizer.fit(np_array_version_of_data)
    binarized_data_array = binarizer.transform(np_array_version_of_data)
    new_binarized_columns = [
        f"binarized_{i}" for i in range(binarized_data_array.shape[1])
    ]
    return pd.DataFrame(binarized_data_array, columns=new_binarized_columns, index=data.index)


def __split_data(
    data: Dataset,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into training and test sets."""
    data_with_extra_features = _add_month_and_year_columns(data)
    x_data = data_with_extra_features.drop(columns=[data.subset_column_name])
    y_data = data_with_extra_features[[data.subset_column_name]]
    
    binarized_x_data = __binarize_data(x_data)

    logging.info("Splitting data into training and test sets...")
    
    return train_test_split(
        binarized_x_data, y_data, test_size=__test_size, shuffle=False
    )


def __create_tsetlin_machine_regression_model(data:Dataset) -> RegressionTsetlinMachine:
    """
    Create a Tsetlin Machine regression model for time series prediction.
    """
    logging.info("Creating Tsetlin Machine regression model...")
    data_targets = data.values[data.subset_column_name]
    number_of_states = int(1 + (data_targets.max() - data_targets.min()) / (2 * __s))
    logging.info(f"Number of states T = {number_of_states}")
    tm = RegressionTsetlinMachine(
        number_of_clauses= __number_of_clauses,
        T=number_of_states,
        s=__s,
        number_of_state_bits=__number_of_state_bits,
        weighted_clauses=True,
    )
    return tm


def __train_tsetlin_machine_regression_model(t_model: Model, x_train: pd.DataFrame, y_train: pd.DataFrame) -> Model:
    """
    Train a Tsetlin Machine regression model.
    """
    #convert to numpy array
    x_train = np.array(x_train)
    #convert to numpy array with 1 dimension
    y_train = np.array(y_train).reshape(-1)
    t_model.fit(x_train, y_train, epochs=__number_of_epochs)
    return t_model


def __predict_with_tsetlin_machine_regression_model(
    model: Model, data: Dataset, x_test: pd.DataFrame, y_test: pd.DataFrame
) -> PredictionData:
    """
    Predict with a Tsetlin Machine regression model.
    """
    title = f"{data.subset_column_name} forecast for {data.subset_row_name} with a Tsetlin Machine Regression Model"
    #convert to numpy array
    x_test_array = np.array(x_test)
    return PredictionData(
        values=model.predict(x_test_array),
        prediction_column_name=None,
        ground_truth_values=y_test,
        confidence_columns=None,
        title=title,
        plot_folder=f"{data.name}/{data.subset_row_name}/tsetlin_machine_regression_model/",
        plot_file_name=f"{data.subset_column_name}_forecast",
    )


tsetlin_machine = method(
    __split_data,
    __create_tsetlin_machine_regression_model,
    __train_tsetlin_machine_regression_model,
    __predict_with_tsetlin_machine_regression_model,
)
