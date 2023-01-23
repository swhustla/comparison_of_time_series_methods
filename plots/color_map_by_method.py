"""Define a color map for each method."""

from typing import Dict


__color_map_by_method_dict = {
        "AR": "red",
        "linear_regression": "green",
        "ARIMA": "orange",
        "HoltWinters": "gray",
        "MA": "indigo",
        "Prophet": "violet",
        "FCNN": "brown",
        "FCNN_embedding": "pink",
        "SARIMA": "darkred",
        "auto_arima": "darkgreen",
        "SES": "darkorange",
        "TsetlinMachine": "lightgray",
    }




def get_color_map_by_method(method_name:str) -> str:
    """Return the color for the method, if present."""
    if method_name in __color_map_by_method_dict:
        return __color_map_by_method_dict[method_name]
    else:
        raise ValueError(f"Method {method_name} not found in color map")
    
