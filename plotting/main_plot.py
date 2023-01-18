#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import TypeVar



Data = TypeVar("Data", contravariant=True)


def load_summary_report() -> Data:
    df = pd.read_csv("../reports/summary_report.csv")
    df["Start Time"] = pd.to_datetime(df["Start Time"],dayfirst=True)
    return df

# print(load_summary_report()["Start Time"].max())

for index, row in load_summary_report().iterrows():
       if row["Dataset"] == "Indian city pollution" and row["Topic"] == 'Ahmedabad':
           print(row["Start Time"],row["Topic"])



# def general_data_selection(methods: list[str], datasets: list[str]):
#     number_of_methods = len(list(methods))
#     for datasetname in datasets:



#     for index, row in load_summary_report().iterrows():
#        if row["Dataset"] == "Indian city pollution" and row["Topic"] == 'Ahmedabad':
#            print(row["Start Time"],row["Topic"])
#Append approprietly 



