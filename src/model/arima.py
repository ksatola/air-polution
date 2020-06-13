import itertools
import pandas as pd
import numpy as np
from logger import logger

from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_model(endog: list, p: int = 0, d: int = 0, q: int = 0):
    """
    Fits SARIMA model using statsmodels SARIMAX function.
    :param endog: data for the model to fit in
    :param p: AR order
    :param d: number of differencing operations to perform
    :param q: MA order
    :return: fitted model object
    """
    # https://stackoverflow.com/questions/54136280/sarimax-python-np-linalg-linalg-linalgerror-lu-decomposition-error
    model = SARIMAX(endog=endog, order=(p, d, q), initialization='approximate_diffuse')
    model_fitted = model.fit()
    return model_fitted


def predict_ts(x: list, coef: list) -> float:
    """
    Forecasts value based on AR() or MA() model (depending if historical observations or error
    residuals are provided as x).
    :param x: historical observations (AR) or error residuals (MA)
    :param coef: constant and auto-regressive/moving average coefficients
    :return: predicted value
    """
    yhat = 0.0
    for i in range(1, len(coef) + 1):
        yhat += coef[i - 1] * x[-i]
    return yhat


def difference(x: list):
    """
    Calculates differences for a list of values.
    :param x: input data
    :return: a list of i-1 differences
    """
    diff = []
    for i in range(1, len(x)):
        value = x[i] - x[i - 1]
        diff.append(value)
    return np.array(diff)


def predict_ar(X: list, coef: list) -> float:
    """
    Forecasts a value using AR() model based on a list of previous values and a list of model
    parameters (as returned by statsmodels models).
    :param X: historical observations
    :param coef: constant and auto-regressive coefficients
    :return: predicted value
    """
    yhat = 0.0
    logger.debug(f'coef -> {coef}')
    logger.debug(f'X -> {X}')
    for i in range(1, len(coef)):
        # X values must be applied backwards as we travel back in the past
        yhat += coef[i] * X[-i]
        logger.debug(coef[i], X[i], yhat)
    return yhat + coef[0]


def get_best_arima_params_for_time_series(data: pd.Series,
                                          seasonal: bool = False,
                                          max_param_range_p: int = 5,
                                          max_param_range_d: int = 2,
                                          max_param_range_q: int = 5) -> ((int, int, int),
                                                                          float):
    """
    Prepares a combinations list of p,d,g ARIMA or SARIMA parameters within specified range and
    searches for their best combination for provided time series data.
    :param data: time series to find best model parameters for
    :param seasonal: if True, SARIMA parameters are searched (seasonal component for ARIMA)
    :param max_param_range_p: maximum value of p parameter used in search
    :param max_param_range_d: maximum value of d parameter used in search
    :param max_param_range_q: maximum value of q parameter used in search
    :return: best found p, d, q parameters of ARIMA
    """
    # Prepares a combinations list of p,d,g ARIMA parameters within specified range
    p = range(0, max_param_range_p + 1)
    d = range(0, max_param_range_d + 1)
    q = range(0, max_param_range_q + 1)
    pdq = list(itertools.product(p, d, q))

    # https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
    # s = 12 for monthly data
    if seasonal:
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    # Search grid best params for the model
    best_res = ()
    best_aic = float('inf')

    # print(seasonal_pdq)

    for param in pdq:

        if seasonal:
            for sparam in seasonal_pdq:
                try:

                    model = SARIMAX(data,
                                    order=param,
                                    seasonal_order=sparam,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
                    results = model.fit()

                    if results.aic < best_aic:
                        print(f'SARIMAX{param}x{sparam} - AIC:{results.aic}')
                        best_res = (param, sparam, results.aic)
                        best_aic = results.aic
                except:
                    continue

        else:
            try:

                model = SARIMAX(data, order=param)
                results = model.fit()

                if results.aic < best_aic:
                    print(f'SARIMAX{param} - AIC:{results.aic}')
                    best_res = (param, results.aic)
                    best_aic = results.aic
            except:
                continue

    if seasonal:
        logger.info(best_res)
        logger.info(f'Best model is SARIMA{best_res[0]}x{best_res[1]} with AIC of {best_aic}')
    else:
        logger.info(f'Best model is ARIMA{best_res[0]} with AIC of {best_aic}')

    return best_res
