"""A module for storing reporting metrics"""
import os
import time
import datetime
import pandas as pd
import platform
import numpy as np


from methods.store_metrics import store_metrics as method



def __write_summary_report(report) -> None:
    """Write a summary report to a file."""
    filepath = "reports/summary_report.csv"
    time_taken = time.time() - report.tstart
    if report.number_of_iterations > 1:
        time_taken = time_taken / report.number_of_iterations
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
            "Elapsed (s)": np.round(time_taken, 4),
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


store_metrics = method(__write_summary_report)