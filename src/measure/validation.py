import pandas as pd
import numpy as np

from model.arima import predict_ar
from measure import get_rmse


def walk_forward_ts_model_validation(data: pd.DataFrame,
                                     col_name: str,
                                     model_params: list,
                                     cut_off_offset: int = 365,
                                     n_pred_points: int = 1,
                                     n_folds: int = -1):
    """
    Validates time series model using time series walk forward model validation method
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