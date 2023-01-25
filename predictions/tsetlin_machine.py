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

The best Python implementation of the Tsetlin Machine I found is the one that is
included in the Tsetlin Machine Python library: 
https://github.com/cair/pyTsetlinMachine

The T parameter is the number of states that the Tsetlin Machine has. In a regression
Tsetlin Machine, with a continuous output, the T parameter is calculated as follows:
T = 1 + (max - min) / 2 * s

where max is the maximum value of the output and min is the minimum value of the output.
s is the number of states per unit of the output. The default value of s is 25,
though a rule of thumb is to use a value of s that is equal to the number of
states that the Tsetlin Machine has.

The grid search is used to find the best parameters for the Tsetlin Machine.

The RegressionTsetlinMachine model in Python has a number of settings as follows:
* number_of_clauses: The number of clauses in the Tsetlin Machine.
* s: The number of states per unit of the output. 
* number_of_features: The number of features in the input pattern. 
* number_of_states: The number of states that the Tsetlin Machine has. 
* weighted_clauses: Whether the clauses are weighted or not. 
* boost_true_positive_feedback: Whether the clauses are boosted or not.
* number_of_epochs: The number of epochs that the Tsetlin Machine is trained for.



"""

from multiprocessing import Pool
from typing import Callable, Tuple, TypeVar, List, Dict, Any

import pandas as pd
import numpy as np

# import tsfresh
# from tsfresh.feature_extraction import EfficientFCParameters

from pyTsetlinMachine.tm import RegressionTsetlinMachine
from pyTsetlinMachine.tools import Binarizer

from methods.tsetlin_machine import tsetlin_machine as method
from methods.tsetlin_machine import tsetlin_machine_single as method_single

from sklearn.model_selection import train_test_split

from data.dataset import Dataset
from predictions.Prediction import PredictionData

from data.seasonal_decompose import seasonal_decompose

from plots.color_map_by_method import get_color_map_by_method

import logging
logging.basicConfig(level=logging.INFO)

import warnings

Model = TypeVar("Model")

__test_size: float = 0.2
__number_of_epochs: int = 100

__number_of_clauses: int = 100
__s: float = 1.9  # s represents the number of states per unit of the output
__number_of_state_bits: int = 2


def __number_of_steps(data: Dataset) -> int:
    return int(len(data.values) * __test_size)

def __get_training_set(data: Dataset) -> Dataset:
    return Dataset(
        name=data.name,
        values=data.values[: -__number_of_steps(data)],
        number_columns=data.number_columns,
        subset_column_name=data.subset_column_name,
        time_unit=data.time_unit,
        subset_row_name=data.subset_row_name,
        seasonality=data.seasonality,
    )


def _add_month_and_year_columns(data: Dataset) -> pd.DataFrame:
    """ Add normalised month and year (and potentially week) columns to a dataframe.
    """
    df = data.values
    # convert to pandas dataframe if series
    if isinstance(df, pd.Series):
        df = df.to_frame()
        df.columns = [data.subset_column_name]

    # Add month and year columns, normalising to start from 0
    df["year"] = df.index.year - df.index.year.min()


    if data.time_unit != "years":
        df["month"] = df.index.month - 1

    if data.time_unit == "days":
        df["week"] = df.index.week - 1
    return df


# def __add_ts_fresh_features_to_data(data: Dataset) -> pd.DataFrame:
#     """
#     Add features to the data using the tsfresh library.
#     This is a library for automated extraction of relevant features from time series.
#     Output is a pandas DataFrame.
#     """
#     logging.info("Adding tsfresh features to data")
#     tsf_settings = EfficientFCParameters()
#     tsf_settings.disable_progressbar = False
#     tsf_settings.n_jobs = 1
#     data_with_features = tsfresh.extract_features(
#         data.values.reset_index(drop=False),
#         column_value=data.subset_column_name,
#         column_sort=data.values.index.name,
#         column_id=data.subset_column_name,
#         default_fc_parameters=tsf_settings,
#     )

    # logging.info(f"Data with features: {data_with_features.head()}")
    # return data_with_features


def __binarize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Binarize the data.

    The Tsetlin Machine requires the data to be binarized,
    i.e. to be represented as a set of 0s and 1s, due to the way that it works.
    """
    logging.debug("Binarizing the data...")
    binarizer = Binarizer(max_bits_per_feature=10)
    np_array_version_of_data = np.array(data)
    binarizer.fit(np_array_version_of_data)
    binarized_data_array = binarizer.transform(np_array_version_of_data)
    new_binarized_columns = [
        f"binarized_{i}" for i in range(binarized_data_array.shape[1])
    ]
    return pd.DataFrame(
        binarized_data_array, columns=new_binarized_columns, index=data.index
    )


def __prepare_training_data(
    x_data: pd.DataFrame, y_data: pd.DataFrame
) -> Tuple[np.array, np.array]:
    """
    Prepare the training data for the Tsetlin Machine.
    """
    logging.debug("Preparing training data...")
    # convert to numpy array
    x_data = np.array(x_data)
    # convert to numpy array with 1 dimension
    y_data = np.array(y_data).reshape(-1)
    return x_data, y_data


def __get_train_and_test_sets(
    x_data: pd.DataFrame, y_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get the training and test sets."""
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=__test_size, shuffle=False
    )
    x_train, y_train = __prepare_training_data(x_train, y_train)
    x_test, y_test = __prepare_training_data(x_test, y_test)
    return x_train, x_test, y_train, y_test


def __prepare_and_split_data(
    data: Dataset,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into training, validation and test sets."""
    logging.debug("Preparing data...")

    data_with_extra_features = _add_month_and_year_columns(data)
    x_data = data_with_extra_features.drop(columns=[data.subset_column_name])

    y_data = data_with_extra_features[[data.subset_column_name]]

    binarized_x_data = __binarize_data(x_data)

    logging.debug("Splitting data into training and test sets...")

    return __get_train_and_test_sets(binarized_x_data, y_data)


def __get_test_date_index(data: Dataset, y_test_shape: int) -> pd.DatetimeIndex:
    """Get the test date index."""
    return data.values.index[-y_test_shape:]

def __create_tsetlin_machine_regression_model(
    data_targets: np.array, config: dict
) -> RegressionTsetlinMachine:
    """
    Create a Tsetlin Machine regression model for time series prediction.
    """
    logging.debug("Creating Tsetlin Machine regression model...")
    number_of_states = int(1 + (data_targets.max() - data_targets.min()) / (2 * __s))
    tm = RegressionTsetlinMachine(
        number_of_clauses=config["number_of_clauses"],
        s=config["s"],
        number_of_state_bits=config["number_of_state_bits"],
        boost_true_positive_feedback=config["boost_true_positive_feedback"],
        T=number_of_states,
        weighted_clauses=config["weighted_clauses"],
    )
    return tm


def __tsetlin_machine_regression_forecast(
    model: RegressionTsetlinMachine,
    x_data: np.ndarray,
    y_data: np.ndarray,
    config: dict,
) -> Tuple[np.array, np.array]:
    """
    Make a forecast using a Tsetlin Machine regression model.
    """
    # fit model
    logging.debug("Fitting Tsetlin Machine regression model...")


    model.fit(
        X=x_data,
        Y=y_data,
        epochs=__number_of_epochs,
        incremental=config["incremental"],
    )
    # make multi-step forecast
    logging.debug("Making multi-step forecast...")

    yhat = model.predict(x_data)
    ground_truth = y_data

    return yhat, ground_truth


def __measure_mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Calculate the mean absolute percentage error."""
    logging.debug("Calculating the mean absolute percentage error...")
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def __validation(x_train: np.array, y_train: np.array, config: dict) -> float:
    """
    Evaluate a config using a given dataset.
    """
    logging.debug("Evaluating config...")
    # create model
    model = __create_tsetlin_machine_regression_model(y_train, config)
    y_pred, y_val = __tsetlin_machine_regression_forecast(model=model, x_data=x_train, y_data=y_train, config=config)
    if type(y_val) == np.array:
        y_val = pd.Series(y_val.flatten())
    # calculate MAPE
    return __measure_mape(actual=y_val, predicted=y_pred)


def __score_model(
    x_train: np.array,
    y_train: np.array,
    config: dict,
    func: Callable[[Dataset, dict], float],
    debug: bool = False,
) -> Tuple[float, dict]:
    """
    Score the model on the test set for a given config.
    """
    result = None
    key = str(config)
    # show all warnings and fail on exception if debugging
    if debug:
        result = func(x_train, y_train, config)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                result = func(x_train, y_train, config)
        except Exception as e:
            logging.error(f"Error during model validation: {e}")
            result = None
            raise e
    # check for an interesting result
    if result is not None:
        print(f" > Model[{key}] MAPE={result:.3f}")
    return (result, config)


def __grid_search(
    x_train: np.array, y_train: np.array, cfg_list: list, parallel: bool = True
) -> List[Tuple[float, dict]]:
    """
    Grid search over configs.
    """
    scores = list()
    # use fork to avoid memory issues with multiprocessing
    # evaluate configs
    if parallel:
        with Pool() as pool:
            scores = pool.starmap(
                __score_model,
                [(x_train, y_train, cfg, __validation) for cfg in cfg_list],
            )
    else:
        scores = [
            __score_model(x_train, y_train, cfg, __validation) for cfg in cfg_list
        ]

    # Remove empty results
    scores = [r for r in scores if r[0] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[0])

    print(f"Printing top 3 config scores:")
    for cfg in scores[:3]:
        print(f" > Model[{cfg[1]}] MAPE={cfg[0]:.3f}")

    return scores


def __tsetlin_configs() -> list:
    """
    Define the configs to be evaluated.
    """
    model_configs = list()
    # define configs
    for s in [2, 2.25, 2.5, 2.75, 3]:
        for number_of_clauses in [250, 500, 1000, 2000, 4000]:
            for number_of_state_bits in [10, 20, 30]:
                for boost_true_positive_feedback in [0, 1]:
                    for incremental in [0, 1]:
                        for weighted_clauses in [False, True]:
                            cfg = {
                                "s": s,
                                "number_of_clauses": number_of_clauses,
                                "number_of_state_bits": number_of_state_bits,
                                "boost_true_positive_feedback": boost_true_positive_feedback,
                                "incremental": incremental,
                                "weighted_clauses": weighted_clauses,
                            }
                            model_configs.append(cfg)
    logging.info(f"Total number of configs: {len(model_configs)}")
    return model_configs

def __seasonal_decompose_data(data: Dataset) -> Dataset:
    """Decomposes the data into trend, seasonal and residual components.
    Return the full decomposition object inside the Dataset object"""

    logging.debug(f"Decomposing {data.name} with STL")

    return seasonal_decompose(data)



def __get_best_model_config(
    data: Dataset, parallel: bool = False
) -> Tuple[dict, int]:
    """
    Get the best model from a grid search.
    """
    # grid search
    configs = __tsetlin_configs()
    total_number_of_configs = len(configs)
    x_train, _ , y_train, _ = __prepare_and_split_data(data)
    scores = __grid_search(x_train=x_train, y_train=y_train, cfg_list=configs, parallel=parallel)
    # get the best config
    best_config = scores[0][1]
    logging.info(f"Best number_of_clauses: {best_config['number_of_clauses']}")
    logging.info(f"Best number_of_state_bits: {best_config['number_of_state_bits']}")
    logging.info(
        f"Best boost_true_positive_feedback: {best_config['boost_true_positive_feedback']}"
    )
    logging.info(f"Best incremental: {best_config['incremental']}")
    logging.info(f"Best weighted_clauses: {best_config['weighted_clauses']}")

    return best_config, total_number_of_configs


def __train_tsetlin_machine_regression_model(
    t_model: RegressionTsetlinMachine, x_train: np.ndarray, y_train: np.ndarray, config: dict
) -> RegressionTsetlinMachine:
    """
    Train a Tsetlin Machine regression model.
    """
    # convert to numpy array with 1 dimension
    y_train = y_train.reshape(-1)
    t_model.fit(x_train, y_train, epochs=__number_of_epochs, incremental=config["incremental"])
    return t_model


def __combine_trend_seasonal_residual(
    trend_prediction: PredictionData,
    seasonal_prediction: PredictionData,
    residual_prediction: PredictionData,
    stl_decomposition_object: Dataset,

) -> PredictionData:
    """
    Combine the trend, seasonal and residual predictions for the test set.
    Update the ground truth values with the original STL decomposition values.
    """
    # get the prediction data object
    combined_prediction = residual_prediction

    # print out the three components in three columns for easy comparison
    combined_prediction_df = pd.concat(
        [trend_prediction.values, seasonal_prediction.values, residual_prediction.values],
        axis=1,
    )
    combined_prediction_df.columns = ["trend", "seasonal", "residual"]
    logging.info(f"Combined prediction components:\n{combined_prediction_df.head()}")

    # compare with seasonal decomposition
    combined_actual_df = pd.concat(
        [stl_decomposition_object.values.trend, stl_decomposition_object.values.seasonal, stl_decomposition_object.values.resid],
        axis=1,
    )
    combined_actual_df.columns = ["trend", "seasonal", "residual"]
    logging.info(f"Combined actual components:\n{combined_actual_df.head()}")

    # combine all components in an additive way
    combined_prediction.values = (
        trend_prediction.values + seasonal_prediction.values + residual_prediction.values
    )


    # get the length of the prediction
    prediction_length = combined_prediction.values.shape[0]

    # update the ground truth values with the original STL decomposition values
    combined_prediction.ground_truth_values = stl_decomposition_object.values.observed[-prediction_length:]
    return combined_prediction


def __get_forecast(
    data: Dataset,
    best_config: dict,
    number_of_configs: int,
) -> PredictionData:
    """
    Get the forecast for the test set.
    """
    title = f"Forecast for {data.name} {data.subset_column_name} using Tsetlin Machine Regression"
    x_train, x_test, y_train, y_test = __prepare_and_split_data(data=data)

    model = __create_tsetlin_machine_regression_model(data_targets=y_train, config=best_config)

    trained_model = __train_tsetlin_machine_regression_model(t_model=model, x_train=x_train, y_train=y_train, config=best_config)

    y_hat, _ = __tsetlin_machine_regression_forecast(model=trained_model, x_data=x_test, y_data=y_test, config=best_config)
 
    return PredictionData(
        method_name="Tsetlin Machine Regression",
        values=pd.Series(y_hat, index=__get_test_date_index(data, y_test.shape[0])),
        prediction_column_name=None,
        ground_truth_values=pd.Series(y_test, index=__get_test_date_index(data, y_test.shape[0])),
        confidence_columns=None,
        title=title,
        plot_folder=f"{data.name}/{data.subset_row_name}/tsetlin_machine_regression_model/",
        plot_file_name=f"{data.subset_column_name}_forecast",
        number_of_iterations=number_of_configs,
        color=get_color_map_by_method("Tsetlin Machine Regression"),
    )



tsetlin_machine = method(
    __seasonal_decompose_data,
    __get_best_model_config,
    __get_forecast,
    __combine_trend_seasonal_residual
)

tsetlin_machine_single = method_single(
    __get_best_model_config,
    __get_forecast,
)
