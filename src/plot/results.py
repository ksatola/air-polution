from measure import prepare_data_for_visualization
import pandas as pd

from logger import logger

from plot import (
    plot_observed_vs_predicted,
    plot_observed_vs_predicted_with_error
)


def visualize_results(show_n_points_of_forecasts: list,
                      start_end_dates: list,
                      plot_types: list,
                      base_file_path: str,
                      fold_results: list,
                      n_pred_points: int,
                      cut_off_offset: int,
                      model_name: str,
                      timestamp: str):
    """

    :param show_n_points_of_forecasts:
    :param start_end_dates:
    :param plot_types:
    :param base_file_path:
    :param fold_results:
    :param n_pred_points:
    :param cut_off_offset:
    :param model_name:
    :return:
    """

    # We need to convert index for plotting from
    # dtype='period[D]' or dtype='period[H]' to dtype='datetime64[ns]'
    # https://stackoverflow.com/questions/29394730/converting-periodindex-to-datetimeindex
    for i in range(0, len(fold_results)):
        if not isinstance(fold_results[i].index, pd.DatetimeIndex):
            fold_results[i].index = fold_results[i].index.to_timestamp()

    k = 0

    # For every n-th point in the future forecast
    for show_n_points_of_forecast in show_n_points_of_forecasts:

        # Build set of datasets
        observed, predicted, error = prepare_data_for_visualization(
            fold_results=fold_results,
            show_n_points_of_forecast=show_n_points_of_forecast,
            n_pred_points=n_pred_points)

        i = 0

        # Create a plot for each start and end dates tuple
        for start_end_date in start_end_dates:

            # Make some space between plots
            print(2 * '\n')

            # Zooming
            start_date = start_end_date[0]
            end_date = start_end_date[1]

            title = f'{model_name} - predictions at lag+{show_n_points_of_forecasts[k]:02}'
            save_path = f'{base_file_path}_{k + 1:02}_lag-{show_n_points_of_forecasts[i]:02}_{timestamp}.png'

            # Plot
            if plot_types[i] == 0:

                plot_observed_vs_predicted(observed=observed[start_date:end_date],
                                           predicted=predicted[start_date:end_date],
                                           num_points=cut_off_offset,
                                           title=title,
                                           label_observed='PM2.5 observed',
                                           label_predicted='PM2.5 predicted',
                                           show_grid=True,
                                           save_path=save_path)
            else:

                plot_observed_vs_predicted_with_error(
                    observed=observed[start_date:end_date],
                    predicted=predicted[start_date:end_date],
                    error=error[start_date:end_date],
                    num_points=cut_off_offset,
                    title=title,
                    label_observed='PM2.5 observed',
                    label_predicted='PM2.5 predicted',
                    label_error='Mean RMSE predition on test dataset',
                    show_grid=True,
                    save_path=save_path)

            logger.info(save_path)
            i += 1
        k += 1
