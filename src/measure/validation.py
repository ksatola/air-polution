import pandas as pd
import numpy as np
from datetime import datetime
from logger import logger

from statsmodels.tsa.statespace.sarimax import SARIMAX

# from model.persistent import (
# ts_naive_forecast,
# sav_fit
# )

from model.arima import (
    predict_ar,
    predict_ts,
    fit_model,
    difference
)

from measure import get_rmse


def sav_fit(data: pd.Series) -> float:
    """
    Simple Average Forecast model.
    :param data: historical data
    :return: average of historical data values
    """
    model_fitted = data.mean()
    return model_fitted


def persistence_fit(data: pd.Series) -> float:
    model_fitted = data[-1:].iloc[0]  # get last value from the series
    return model_fitted


def walk_forward_ref_model_validation(data: pd.DataFrame,
                                      col_name: str,
                                      model_type: str,
                                      cut_off_offset: int = 365,
                                      n_pred_points: int = 1,
                                      n_folds: int = -1,
                                      period: str = '',
                                      rolling_window: int = 4
                                      ):
    """
    XXXXXXXXXXXXX SAV
    Validates time series using SARIMA family of models using time series walk forward model validation method
    :param data: pandas DataFrame
    :param col_name: time series column name
    :param model_type: 'SAV'
    :param cut_off_offset: number of observation point counted backwards from the last newest observation
    :param n_pred_points: number of forecast points (used in each fold)
    :param n_folds: if -1 number of folds equals all test data points, otherwise number of folds
    counted backwards from the last / newest observation
    :param period: SARIMA models require index to be of PeriodIndex rather than DatetimeIndex type
    :return: list of data frames with results from each fold
    """

    # Take entire dataset and split it to train/test
    # according to train_test_split_position using cut_off_offset
    train_test_split_position = int(len(data) - cut_off_offset)
    max_n_folds = len(data) - cut_off_offset - n_pred_points

    # print(f'len(data) -> {len(data)}')
    # print(f'cut_off_offset -> {cut_off_offset}')
    # print(f'train_test_split_position -> {train_test_split_position}')
    # print(f'n_pred_points -> {n_pred_points}')
    # print(f'max_n_folds -> {max_n_folds}')

    if n_folds < 1:
        last_n_folds_pos = len(data) - n_pred_points
    else:
        if n_folds >= max_n_folds:
            last_n_folds_pos = max_n_folds - 1
        # else:
        # last_n_folds_pos = train_test_split_position + n_folds

    # print(f'last_n_folds_pos -> {last_n_folds_pos}')

    # A list of data frames with results from each fold n_pred_points predictions
    fold_results = []

    # Setup frequency (H or D) - required by models working on data-indexed points
    # if period:
    # data.index = pd.DatetimeIndex(data.index).to_period(period)

    fold_nums = cut_off_offset  # last_n_folds_pos - train_test_split_position
    fold_num = 0
    dt_format = "%Y-%m-%d_%H-%M-%S"

    # Frequency of displaying diagnostic information about validation process
    if cut_off_offset >= 500 and cut_off_offset < 1000:
        divider = 20
    elif cut_off_offset >= 1000 and cut_off_offset < 10000:
        divider = 100
    elif cut_off_offset >= 10000:
        divider = 1000
    else:
        divider = 10

    # logger.info(f'({order[0]}, {order[1]}, {order[2]})')
    if model_type == 'SAF':
        logger.info(f'SAF model validation started')
    elif model_type == 'PER':
        logger.info(f'PER model validation started')
    elif model_type == 'SMA':
        logger.info(f'SMA model validation started')
    elif model_type == 'EMA':
        logger.info(f'EMA model validation started')
    else:
        logger.info(f'Exit: no model type provided')

    # For each data point in the test part
    for i in range(train_test_split_position, last_n_folds_pos):

        # print(f'i -> {i}')
        # print(f'train_test_split_position -> {train_test_split_position}')
        # print(f'last_n_folds_pos -> {last_n_folds_pos}')

        # Show progress indicator (every divider folds)
        if fold_num % divider == 0:
            print(f'Started fold {fold_num:06}/{fold_nums:06} - '
                  f'{datetime.now().strftime(dt_format)}')
        fold_num += 1

        # For each fold
        history = data[0:i].copy()
        future = data[i: i + n_pred_points].copy()

        # print(f'history.shape {history.shape}')
        # print(f'future.shape {future.shape}')
        # print(history.head(10))
        # print(future.head(10))

        predicted = []

        # Forecast values for n_pred_points (for example: 7-days forecast)
        for j in range(len(future)):
            # print(history.tail(5))

            # Fit a model with updated historical data (each time we add a predicted value to the end of history)
            # https://stackoverflow.com/questions/54136280/sarimax-python-np-linalg-linalg-linalgerror-lu-decomposition-error
            # model = SARIMAX(endog=history, order=(p, d, q), initialization='approximate_diffuse')
            # model_fitted = fit_model(endog=history, p=p, d=d, q=q)

            # print(f'j -> {j}')
            # print(f'type(history) -> {type(history)}')
            # print(f'history.tail(5) -> {history.tail(5)}')
            # print(f'model_fitted -> {model_fitted}')

            # Get prediction for t+j lag
            # SAF
            if model_type == 'SAF':
                model_fitted = sav_fit(history[col_name])
                yhat = model_fitted  # just history.mean()

            # PER
            elif model_type == 'PER':
                model_fitted = persistence_fit(history[col_name])
                yhat = model_fitted  # just last history element

            # SMA
            elif model_type == 'SMA':
                yhat = history[col_name].rolling(window=rolling_window).mean()[-1:][0] # moving average

            # EMA
            elif model_type == 'EMA':
                # Get the moving average (Exponentially-weighted-window) for last element of train
                # https://pandas.pydata.org/pandas-docs/stable/user_guide/computation.html#exponentially-weighted-windows
                yhat = history[col_name].ewm(span=30).mean()[-1:][0]  # exponentially moving
                # average

            # model MA -> order = (0, 0, x)
            # elif model_type == 'MA':
            # ma_coef = model_fitted.maparams
            # residuals = model_fitted.resid
            # yhat = predict_ts(residuals, ma_coef)

            # model ARMA -> order = (x, 0, y)
            # elif model_type == 'ARMA':
            # ar_coef = model_fitted.arparams
            # ma_coef = model_fitted.maparams
            # residuals = model_fitted.resid
            # history_as_list = history[col_name].tail(len(ar_coef) + 1).tolist()
            # yhat = predict_ts(history_as_list, ar_coef) + predict_ts(residuals, ma_coef)

            # model ARIMA -> order = (x, y, z)
            # elif model_type == 'ARIMA':
            # ar_coef = model_fitted.arparams
            # ma_coef = model_fitted.maparams
            # residuals = model_fitted.resid
            # history_as_list = history[col_name].tail(len(ar_coef) + 1).tolist()
            # diff = history_as_list
            # for k in range(d):
            # diff = difference(diff)
            # yhat = history_as_list[-1] + predict_ts(history_as_list, diff) + predict_ts(
            # residuals, ma_coef)

            else:
                pass

            # Add prediction value to results
            predicted.append(yhat)

            # print(f'before: history.tail(2) -> {history.tail(2)}')

            # Create a new row with the next data point index from future
            # Extend history with the last predicted value (we need an index of this value)
            history = history.append(future[j: j + 1])

            # Replace observed value with predicted value (update data point value at newly created index)
            history.loc[future[j: j + 1].index] = [yhat]

            # print(f'after: history.tail(2) -> {history.tail(2)}')

            # if (j > 2): return

        # Summarize results for the fold
        # Each row represents next predicted lag
        df_fold_observed = future[col_name].copy()  # observed
        df_fold_predicted = history[-n_pred_points:].copy()  # predicted
        df_fold_results = pd.concat([df_fold_observed, df_fold_predicted], axis=1)
        df_fold_results.columns = ['observed', 'predicted']

        df_fold_results['error'] = np.abs \
            (df_fold_results['observed'] - df_fold_results['predicted'])  # error
        df_fold_results['abs_error'] = np.abs \
            (df_fold_results['observed'] - df_fold_results['predicted'])  # absolute error

        fold_results.append(df_fold_results)
        # print(df_fold_results)

    return fold_results


def walk_forward_ts_model_validation2(data: pd.DataFrame,
                                      col_name: str,
                                      model_type: str,
                                      p: int = 0,
                                      d: int = 0,
                                      q: int = 0,
                                      cut_off_offset: int = 365,
                                      n_pred_points: int = 1,
                                      n_folds: int = -1,
                                      period: str = ''):
    """
    Validates time series using SARIMA family of models using time series walk forward model validation method
    :param data: pandas DataFrame
    :param col_name: time series column name
    :param model_type: 'AR', 'MA', 'ARMA', 'ARIMA', 'SARIMA'
    :param cut_off_offset: number of observation point counted backwards from the last newest observation
    :param n_pred_points: number of forecast points (used in each fold)
    :param n_folds: if -1 number of folds equals all test data points, otherwise number of folds
    counted backwards from the last / newest observation
    :param period: SARIMA models require index to be of PeriodIndex rather than DatetimeIndex type
    :return: list of data frames with results from each fold
    """

    order = (p, d, q)

    # Take entire dataset and split it to train/test
    # according to train_test_split_position using cut_off_offset
    train_test_split_position = int(len(data) - cut_off_offset)
    max_n_folds = len(data) - train_test_split_position

    if n_folds < 1:
        last_n_folds_pos = len(data)
    else:
        if n_folds > max_n_folds:
            last_n_folds_pos = max_n_folds
        else:
            last_n_folds_pos = train_test_split_position + n_folds

    # print(range(train_test_split_position, n_folds))

    # A list of data frames with results from each fold n_pred_points predictions
    fold_results = []

    # Setup frequency (H or D) - required by models working on data-indexed points
    # if period:
    # data.index = pd.DatetimeIndex(data.index).to_period(period)

    fold_nums = last_n_folds_pos - train_test_split_position
    fold_num = 0
    dt_format = "%Y-%m-%d_%H-%M-%S"

    # Frequency of displaying diagnostic information about validation process
    if cut_off_offset >= 500 and cut_off_offset < 1000:
        divider = 20
    elif cut_off_offset >= 1000 and cut_off_offset < 10000:
        divider = 100
    elif cut_off_offset >= 10000:
        divider = 1000
    else:
        divider = 10

    logger.info(f'({order[0]}, {order[1]}, {order[2]})')
    if model_type == 'AR':
        logger.info(f'AR model validation started')
    elif model_type == 'MA':
        logger.info(f'MA model validation started')
    elif model_type == 'ARMA':
        logger.info(f'ARMA model validation started')
    elif model_type == 'ARIMA':
        logger.info(f'ARIMA model validation started')
    # elif model_type == 'SARIMA':
    # logger.info(f'SARIMA model validation started')
    else:
        logger.info(f'Exit: no model type provided')
        return

    # For each data point in the test part
    for i in range(train_test_split_position, last_n_folds_pos):

        # Show progress indicator (every divider folds)
        if fold_num % divider == 0:
            print(f'Started fold {fold_num:06}/{fold_nums:06} - '
                  f'{datetime.now().strftime(dt_format)}')

        fold_num += 1

        # For each fold
        history = data[0:i].copy()
        future = data[i: i + n_pred_points].copy()

        # print(f'history.shape {history.shape}')
        # print(f'future.shape {future.shape}')
        # print(history.head(1))
        # print(future.head(10))

        predicted = []

        # Forecast values for n_pred_points
        for j in range(len(future)):
            # print(history.tail(5))

            # Fit a model with updated historical data (each time we add a predicted value to the end of history)
            # https://stackoverflow.com/questions/54136280/sarimax-python-np-linalg-linalg-linalgerror-lu-decomposition-error
            # model = SARIMAX(endog=history, order=(p, d, q), initialization='approximate_diffuse')
            model_fitted = fit_model(endog=history, p=p, d=d, q=q)

            # Get prediction for t+1 lag
            # model AR -> order = (x, 0, 0)
            if model_type == 'AR':
                ar_coef = model_fitted.arparams
                history_as_list = history[col_name].tail(len(ar_coef) + 1).tolist()
                yhat = predict_ts(history_as_list, ar_coef)

            # model MA -> order = (0, 0, x)
            elif model_type == 'MA':
                ma_coef = model_fitted.maparams
                residuals = model_fitted.resid
                yhat = predict_ts(residuals, ma_coef)

            # model ARMA -> order = (x, 0, y)
            elif model_type == 'ARMA':
                ar_coef = model_fitted.arparams
                ma_coef = model_fitted.maparams
                residuals = model_fitted.resid
                history_as_list = history[col_name].tail(len(ar_coef) + 1).tolist()
                yhat = predict_ts(history_as_list, ar_coef) + predict_ts(residuals, ma_coef)

            # model ARIMA -> order = (x, y, z)
            elif model_type == 'ARIMA':
                ar_coef = model_fitted.arparams
                ma_coef = model_fitted.maparams
                residuals = model_fitted.resid
                history_as_list = history[col_name].tail(len(ar_coef) + 1).tolist()
                diff = history_as_list
                for k in range(d):
                    diff = difference(diff)
                    yhat = history[col_name].shift() \
                           + predict_ts(history_as_list, diff) \
                           + predict_ts(residuals, ma_coef)

            else:
                pass

            # Add prediction value to results
            predicted.append(yhat)

            # Create a new row with the next data point index from future
            # Extend history with the last predicted value (we need an index of this value)
            history = history.append(future[j: j + 1])

            # Replace observed value with predicted value (update data point value at newly created index)
            history.loc[future[j: j + 1].index] = [yhat]

        # Summarize results for the fold
        # Each row represents next predicted lag
        df_fold_observed = future[col_name].copy()  # observed
        df_fold_predicted = history[-n_pred_points:].copy()  # predicted
        df_fold_results = pd.concat([df_fold_observed, df_fold_predicted], axis=1)
        df_fold_results.columns = ['observed', 'predicted']

        df_fold_results['error'] = np.abs \
            (df_fold_results['observed'] - df_fold_results['predicted'])  # error
        df_fold_results['abs_error'] = np.abs \
            (df_fold_results['observed'] - df_fold_results['predicted'])  # absolute error

        fold_results.append(df_fold_results)
        # print(df_fold_results)

    return fold_results


def walk_forward_ts_model_validation(data: pd.DataFrame,
                                     col_name: str,
                                     model_params: list,
                                     cut_off_offset: int = 365,
                                     n_pred_points: int = 1,
                                     n_folds: int = -1):
    """
    Validates time series AR model using time series walk forward model validation method.
    :param data: pandas DataFrame
    :param col_name: time series column name
    :param model_params: model params returned by statsmodels estimator (AR)
    :param cut_off_offset: number of observation point counted backwards from the last
    / newest observation
    :param n_pred_points: number of forecast points (used in each fold)
    :param n_folds: if -1 number of folds equals all test data points, otherwise number of folds
    counted backwards from the last / newest observation
    :return: list of data frames with results from each fold
    """

    # Take entire dataset and splits it to train/test
    # according to train_test_split_position using cut_off_offset
    train_test_split_position = int(len(data) - cut_off_offset)
    max_n_folds = len(data) - train_test_split_position

    if n_folds < 1:
        last_n_folds_pos = len(data)
    else:
        if n_folds > max_n_folds:
            last_n_folds_pos = max_n_folds
        else:
            last_n_folds_pos = train_test_split_position + n_folds

    # print(range(train_test_split_position, n_folds))

    # A list of data frames with results from each fold n_pred_points predictions
    fold_results = []

    # Do this for each data point in the test part
    for i in range(train_test_split_position, last_n_folds_pos):

        # For each fold
        history = data[0:i].copy()
        future = data[i: i + n_pred_points].copy()

        # print(f'history.shape {history.shape}')
        # print(f'future.shape {future.shape}')
        # print(history.head(1))
        # print(future.head(10))

        predicted = []

        # Forecast values for n_pred_points
        for j in range(len(future)):
            # print(history.tail(5))

            # Get prediction for t+1 lag
            yhat = predict_ar(history[col_name].tail(len(model_params) + 1).tolist(), model_params)
            # print(list(reversed(history[col_name].tail().tolist())))

            # history[col_name].tail(len(model_params)).tolist()

            # Add it as a predicted value
            predicted.append(yhat)
            # print(predicted)

            # Create a new row with the next data point index from df_test
            # Extend history with the last predicted value
            history = history.append(future[j: j + 1])

            # Replace observed value with predicted value
            history.loc[future[j: j + 1].index] = [yhat]

        # Summarize results for the fold
        # Each row represents next predicted lag
        df_fold_observed = future[col_name].copy()  # observed
        df_fold_predicted = history[-n_pred_points:].copy()  # predicted
        df_fold_results = pd.concat([df_fold_observed, df_fold_predicted], axis=1)
        df_fold_results.columns = ['observed', 'predicted']

        df_fold_results['error'] = np.abs \
            (df_fold_results['observed'] - df_fold_results['predicted'])  # error
        df_fold_results['abs_error'] = np.abs \
            (df_fold_results['observed'] - df_fold_results['predicted'])  # absolute error

        fold_results.append(df_fold_results)
        # print(df_fold_results)

    return fold_results


def get_mean_folds_rmse_for_n_prediction_points(fold_results: list, n_pred_points: int = 1):
    # For each fold and number of prediction points calculate rmse
    # Returns a list of mean folds RMSE for n_pred_points (starting at 1)

    mean_rmse_for_prediction_points = []

    # Fo each number od prediction points
    for i in range(1, n_pred_points + 1):

        show_n_points_of_forecast = i
        start_index = show_n_points_of_forecast - 1
        end_index = show_n_points_of_forecast

        rmse_for_folds = []

        # For each fold
        # n_pred_points results from the end contain NaNs
        for fold in fold_results[:len(fold_results) - n_pred_points]:
            rmse = get_rmse(observed=fold[start_index:end_index]['observed'],
                            predicted=fold[start_index:end_index]['predicted'])
            rmse_for_folds.append(rmse)
            # print(f'{i:03} = {model_name} RMSE {rmse}')

        # Calculate average RMSE for a prediction point
        mean_rmse_for_prediction_point = pd.Series(rmse_for_folds).mean()
        mean_rmse_for_prediction_points.append(mean_rmse_for_prediction_point)

    return mean_rmse_for_prediction_points


def prepare_data_for_visualization(fold_results: list,
                                   show_n_points_of_forecast: int,
                                   n_pred_points: int) -> (
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame):
    """
    Prepares 3 data series (observed, predicted, error) from a specified forecast interval (
    number of predicted point).
    :param fold_results: list of dataframes with validation results
    :param show_n_points_of_forecast: specifies which point predicted in the future should be
    returned for all folds (only this point)
    :param n_pred_points: number of forecasted points in a fold
    :return: tuple of 3 pandas DataFrames with observed, predicted and error values
    """
    start_index = show_n_points_of_forecast - 1
    end_index = show_n_points_of_forecast

    observed = pd.Series()
    predicted = pd.Series()
    error = pd.Series()

    # n_pred_points results from the end contain NaNs
    for fold in fold_results[:len(fold_results) - n_pred_points]:
        observed = observed.append(fold[start_index:end_index]['observed'])
        predicted = predicted.append(fold[start_index:end_index]['predicted'])
        error = error.append(fold[start_index:end_index]['error'])

    return observed, predicted, error
