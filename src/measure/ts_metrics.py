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
    Calculates MAE - Mean Absolute Error between two series as percentage change
    :param observed: pandas series with observed values (the truth)
    :param predicted: pandas series with predicted values
    :return: MAE value
    """
    return round(abs(observed.pct_change() - predicted.pct_change()).mean(), 4)


def get_hit_rate(observed: pd.Series, predicted: pd.Series) -> np.float:
    """
    Calculates "hit rate" between two series. The hit is defined as the same decision regarding
    prediction direction comparing to observed direction (time series next point going up or down).
    :param observed: pandas series with observed values (the truth)
    :param predicted: pandas series with predicted values
    :return: MAE value
    """
    # Build a data frame
    pred_pct = pd.concat([observed.pct_change(), predicted.pct_change()], axis=1)
    pred_pct.dropna(inplace=True)
    pred_pct.columns = ['Observed', 'Predicted']

    # Mark rows with a move in the same direction
    pred_pct['Hit'] = np.where(np.sign(pred_pct['Observed']) == np.sign(pred_pct['Predicted']), 1, 0)

    # Returm the hit rate
    return round((pred_pct['Hit'].sum() / pred_pct['Hit'].count()) * 100, 2)
