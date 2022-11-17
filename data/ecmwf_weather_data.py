"""Load in ECWMF weather data."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, TypeVar

import pandas as pd

from ecmwfapi import ECMWFDataServer

Data = TypeVar("Data", contravariant=True)
from data.dataset import Dataset
from data.impute_data import impute
import configparser

logger = logging.getLogger(__name__)


def set_environment_variables() -> configparser.ConfigParser:
    """Set environment variables for the ECMWF API from the file $HOME/.ecmwfapirc."""
    # load in the environment variables
    config = configparser.ConfigParser()
    config.read(os.path.expanduser("~/.ecmwfapirc"))
    print(config.sections())
    os.environ["ECMWF_URL"] = config["ECMWF_API"]["url"]
    os.environ["ECMWF_API_KEY"] = config["ECMWF_API"]["key"]
    os.environ["ECMWF_API_EMAIL"] = config["ECMWF_API"]["email"]

    return config

    

def __download_if_needed(lat_long:Tuple[str, str, str, str], time_frame:Tuple[str, str]) -> Path:
    """Download weather data for given lat long and timeframe
    if it is not already downloaded.
    We need boundary layer height and radiation data / cloud cover data (ECMWF).
    """
    path = Path("data/weather_data")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        os.chdir(path)

        config = __set_environment_variables()
        server = ECMWFDataServer(url=config["url"], key=config["key"], email=config["email"])

        server.retrieve({
            "class": "ei",
            "dataset": "interim",
            "date": "/TO/".join([str(date) for date in time_frame]),
            "expver": "1",
            "grid": "0.75/0.75",
            "levtype": "sfc",
            "param": "134.128/165.128/166.128/167.128",
            "step": "0",
            "stream": "oper",
            "time": "00:00:00",
            "type": "an",
            "format": "netcdf",
            "target": "weather_data.nc",
            "area": f"{lat_long[0]}/{lat_long[1]}/{lat_long[2]}/{lat_long[3]}",
        })
        
        # Download data from ECMWF
        # https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets
        # https://confluence.ecmwf.int/display/WEBAPI/Python+ERA5+examples
        # https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets#AccessECMWFPublicDatasets-ExampleofretrievingERA5data
        # https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5
        # https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5#HowtodownloadERA5-ExampleofretrievingERA5data
