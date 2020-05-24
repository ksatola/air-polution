import pandas as pd

from plot import (
    plot_train_test_predicted,
    plot_observed_vs_predicted,
    plot_observations_to_predictions_relationship
)


def ts_naive_forecast(data: pd.Series,
                      train_test_split_ratio: float = 0.8,
                      num_points_test_split: int = 0,
                      show_plots: bool = True) -> pd.DataFrame:
    """
    Builds persistence model for provided time series with specified train-test split ratio or
    number of points counted backward starting with the last observation. Optionally creates
    diagnostic plots.
    :param data: pandas Series with observed data
    :param train_test_split_ratio: between 0 to 1, used if num_points_split is greater than 0
    :param num_points_test_split: number of points counted backward starting with the last
    observation treated as test data
    :param show_plots: if True, shows data plot for train, test and predicted parts of the
    entire time series, zoomed plot of observed vs. predicted data and predictions vs.
    observations plot.
    :return: pandas DataFrame with shifted data and predictions for every time series test point
    """

    if train_test_split_ratio < 0 or train_test_split_ratio > 1:
        train_test_split_ratio = 0.7

    # Prepare data frame
    df_persistent_model = pd.concat([data.shift(1), data], axis=1)
    df_persistent_model.columns = ['t', 't+1']

    # Remove the first row (with a NaN)
    df_persistent_model = df_persistent_model.iloc[1:]

    # Split data into train and test data sets
    if num_points_test_split <= 0:
        train_size = int(len(data) * train_test_split_ratio)
    else:
        train_size = int(len(data) - num_points_test_split - 1)
        if train_size < 0: train_size = len(data)

    df_train, df_test = df_persistent_model[0:train_size], df_persistent_model[train_size:]

    # Predict
    predicted = df_test.copy()
    predicted['pred'] = predicted['t']
    print(f'Number of predicted points: {predicted.shape[0]}')

    if show_plots:
        # Plot result
        plot_train_test_predicted(train=df_train['t+1'],
                                  test=df_test['t+1'],
                                  predicted=predicted['pred'],
                                  title="Naive Forecast - Train Test Split with Predictions")

        plot_observed_vs_predicted(observed=df_test['t+1'],
                                   predicted=predicted['pred'],
                                   num_points=predicted.shape[0],
                                   title="Naive Forecast - Observed vs. Predicted",
                                   label_observed='PM2.5 observed',
                                   label_predicted='PM2.5 predicted')

        plot_observations_to_predictions_relationship(observed=df_test['t+1'],
                                                      predicted=predicted['pred'],
                                                      title="Naive Forecast - Predicted vs"
                                                            "Observed Percentage Change",
                                                      label_observed='Observations',
                                                      label_predicted='Predictions')

    return predicted
