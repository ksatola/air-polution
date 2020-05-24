import itertools
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX


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
    p = range(0, max_param_range_p+1)
    d = range(0, max_param_range_d+1)
    q = range(0, max_param_range_q+1)
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
        print(best_res)
        print(f'Best model is SARIMA{best_res[0]}x{best_res[1]} with AIC of {best_aic}')
    else:
        print(f'Best model is ARIMA{best_res[0]} with AIC of {best_aic}')

    return best_res
