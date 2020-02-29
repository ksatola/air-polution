import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def get_rmse(observed: pd.Series, predicted: pd.Series) -> np.float:
    """
    Calculates RMSE - Root Mean Squared Error between two series
    :param observed: pandas series with observed values (the truth)
    :param predicted: pandas series with predicted values
    :return: RMSE value
    """
    mse = mean_squared_error(observed, predicted)
    rmse = round(np.sqrt(mse), 4)
    return rmse


def get_mae(observed: pd.Series, predicted: pd.Series) -> np.float:
    """
    Calculates MAE - Mean Absolute Error between two series
    :param observed: pandas series with observed values (the truth)
    :param predicted: pandas series with predicted values
    :return: MAE value
    """
    return round(abs(observed.pct_change() - predicted.pct_change()).mean(), 4)
