"""Load in ECWMF weather data.


ECMWF is the European Centre for Medium-Range Weather Forecasts.
This data is accessed via the ECMWF API, and is freely available.
A key, email and url are required to access the data.
Example data types are boundary layer height, radiation data, cloud cover data and more.

"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, TypeVar

import pandas as pd

from ecmwfapi import ECMWFDataServer, cdsapi

Data = TypeVar("Data", contravariant=True)
from data.dataset import Dataset
from data.impute_data import impute
import configparser

logger = logging.getLogger(__name__)

# TODO: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview
# https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5#HowtodownloadERA5-ExampleofretrievingERA5data
# For land data:

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

      # Blaise's creds
#   wf_set_key(user = "59954",
#              key = "ea80fa47-cd4a-4fd5-a7ef-6a270be2a5e4",
#              service = "cds")

def __download_if_needed(lat_long:Tuple[str, str, str, str], time_frame:Tuple[str, str]) -> Path:
    """Download weather data for given lat long and timeframe
    if it is not already downloaded.
    We need boundary layer height and radiation data / cloud cover data (ECMWF).
    """
    path = Path("data/weather_data")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        os.chdir(path)
        # https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form
        config = __set_environment_variables()
        cds_client = cdsapi.Client(
            key=config["CDS_API"]["key"],
            email=config["CDS_API"]["email"],
            service=config["CDS_API"]["service"],
        )

        cds_client.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': [
                    '2m_temperature', 'total_precipitation', '10m_u_component_of_wind',
                    '10m_v_component_of_wind', 'surface_pressure',
                ],
                'year': ['2019', '2020'],
                'month': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'],
                'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
                'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00', '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
                'format': 'netcdf',
            },
            'download.nc')
        os.chdir("..")
        
        # Download data from ECMWF
        # https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets
        # https://confluence.ecmwf.int/display/WEBAPI/Python+ERA5+examples
        # https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets#AccessECMWFPublicDatasets-ExampleofretrievingERA5data
        # https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5
        # https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5#HowtodownloadERA5-ExampleofretrievingERA5data
