"""Impute data using a given method."""

import pandas as pd

from data.dataset import Dataset



def impute(data: pd.DataFrame) -> pd.DataFrame:
    """Impute the data using the given method."""
    return data