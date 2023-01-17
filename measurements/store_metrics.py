"""A module for storing reporting metrics"""
import os
import time
import datetime
import pandas as pd
import platform
import numpy as np

import logging

import json
import gzip

from data.report import Report

from methods.store_metrics import store_metrics as method



def __write_summary_report(report) -> None:
    """Write a summary report to a file."""
    filepath = "reports/summary_report.csv"
    time_taken = time.time() - report.tstart
    if report.prediction.number_of_iterations > 1:
        time_taken = time_taken / report.prediction.number_of_iterations
    this_summary_dataframe = pd.DataFrame(
        {
            "Start Time": datetime.datetime.fromtimestamp(report.tstart).strftime(
                "%d-%m-%Y %H:%M:%S"
            ),
            "Platform": platform.platform(),
            "Dataset": report.dataset.name,
            "Topic": report.dataset.subset_row_name,
            "Feature": report.dataset.subset_column_name,
            "Model": report.method,
            "RMSE": report.metrics["root_mean_squared_error"],
            "R Squared": report.metrics["r_squared"],
            "MAE": report.metrics["mean_absolute_error"],
            "MAPE": report.metrics["mean_absolute_percentage_error"],
            "Elapsed (s)": np.round(time_taken, 4),
            "Filepath": report.filepath,
        },
        index=[0],
    )
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if os.path.exists(filepath):
        previous_summary_dataframe = pd.read_csv(filepath, index_col=None)
    else:
        previous_summary_dataframe = pd.DataFrame()


    this_summary_dataframe = pd.concat(
        [previous_summary_dataframe, this_summary_dataframe]
    )

    this_summary_dataframe.to_csv(filepath, index=False)


def __store_report_to_file(report: Report) -> None:
    """Store a prediction object from report to a JSON file."""
    os.makedirs(os.path.dirname(report.filepath), exist_ok=True)

    # convert the prediction object to a JSON string, without using to_json()
    # handle the case where the one of the prediction object attributes is a Pandas DataFrame
    # or Timestamp, or any case where the attribute is not JSON serializable

    dict_for_json = {}

    # convert all keys to one of the following types: str, int, float, bool or None
    for key, value in report.prediction.__dict__.items():
        if isinstance(value, pd.DataFrame):
            dict_for_json[key] = str(value.to_dict())
            
        elif isinstance(value, pd.Timestamp):
            dict_for_json[key] = value.strftime("%Y-%m-%d %H:%M:%S")

        elif isinstance(value, np.ndarray):
            dict_for_json[key] = value.tolist()
        elif isinstance(value, pd.Series):
            dict_for_json[key] = str(value.to_dict())

        # add more cases here if needed, avoid too many elifs

        elif type(value) in [dict, int, float, str, bool, type(None), list]:
            dict_for_json[key] = value

        else:
            logging.debug(f"Could not convert {key} to JSON. Skipping.")
            continue
        print(f"Type of [{key}]={value} is {type(dict_for_json[key])}")
        
    json_string = json.dumps(dict_for_json, indent=4)
    # compress the file when saving to disk
    with gzip.open(report.filepath, "wt") as file:
        file.write(json_string)
        


store_metrics = method(__write_summary_report, __store_report_to_file)