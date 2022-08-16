from methods.ARIMA import arima as method

from arch.unitroot import KPSS, ADF
from pmdarima.arima.utils import ndiffs
from statsmodels.tsa.forecasting.stl import STLForecast
import statsmodels.api as sm

def __stationarity(data):
    return ADF(data).pvalue < 0.05

def __get_differencing_term(data):
    return ndiffs(data, test="adf")

def __fit_auto_regressive_model(data):
    model = STLForecast(data,
    sm.tsa.arima.ARIMA,
    model_kwargs=dict(order=(1, __get_differencing_term(data), 0), trend="t"),
    )
    return model.fit().model_result

def __forecast(model, number_of_steps):
    return model.get_forecast(steps=number_of_steps).summary_frame()


def __number_of_steps(data):
    return len(data) // 10


arima = method(__stationarity, __number_of_steps, __fit_auto_regressive_model, __forecast)