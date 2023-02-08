import unittest

import pandas as pd

from reports.report_loader import report_loader, json_report_loader
from predictions.Prediction import PredictionData

ZIPPED_JSON_FILENAME = "reports/full_data/Airline passengers_all_AR_2023-01-19.json.gz"


# to run this test, run the following command in the terminal:
# python -m unittest tests/test_json_to_predictiondata_object.py


class TestReportLoader(unittest.TestCase):
    @staticmethod
    def test_report_loader():
        """Test that the report loader returns a dataframe."""
        report = report_loader()
        assert isinstance(report, pd.DataFrame)

    @staticmethod
    def test_report_loader_not_empty():
        """Test that the report loader returns a dataframe."""
        report = report_loader()
        assert not report.empty

    @staticmethod
    def test_report_loader_columns():
        """Test that the report loader returns a dataframe."""
        report = report_loader()

        assert report.columns.tolist() == [
            "Platform",
            "Dataset",
            "Topic",
            "Feature",
            "Model",
            "RMSE",
            "R Squared",
            "MAE",
            "Elapsed (s)",
            "MAPE",
            "Filepath",
        ]

    @staticmethod
    def test_json_to_predictiondata():
        """Test that the json file can be loaded into a PredictionData object."""
        prediction_data_object = json_report_loader(ZIPPED_JSON_FILENAME)
        assert isinstance(prediction_data_object, PredictionData)

    @staticmethod
    def test_json_to_predictiondata_keys():
        """Test that the json file can be loaded into a PredictionData object.
        The keys of a PredictionData object should match the keys of the json file."""
        prediction_data_object = json_report_loader(ZIPPED_JSON_FILENAME)

        assert prediction_data_object.__dict__.keys() == {
            "method_name",
            "values",
            "prediction_column_name",
            "ground_truth_values",
            "confidence_columns",
            "title",
            "plot_folder",
            "plot_file_name",
            "model_config",
            "number_of_iterations",
            "confidence_on_mean",
            "confidence_method",
            "color",
            "in_sample_prediction",
        }


if __name__ == "__main__":
    unittest.main()
