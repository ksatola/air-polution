import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import DecomposeResult
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_train_test_predicted(train: pd.Series,
                              test: pd.Series,
                              predicted: pd.Series,
                              title: str = 'Train, Test and Predicted',
                              label_train: str = 'Train',
                              label_test: str = 'Test',
                              label_predicted: str = 'Predicted') -> None:
    """
    Plots entire time series with train, test split and predicted values.
    :param train: train part of the time series
    :param test: test part of the time series
    :param predicted: predicted values
    :param title: plot title
    :param label_train: label for the train data
    :param label_test: label for the test data
    :param label_predicted: label for the predicted data
    :return: None
    """
    plt.figure(figsize=(20, 10))
    plt.plot(train.index, train, label=label_train, c='blue')
    plt.plot(test.index, test, label=label_test, c='orange')
    plt.plot(predicted.index, predicted, label=label_predicted, c='green')
    plt.title(title)
    plt.legend(loc='best')
    plt.show()


def plot_observed_vs_predicted(observed: pd.Series,
                               predicted: pd.Series,
                               num_points: int,
                               title: str = 'Observed vs. Predicted',
                               label_observed: str = 'Observed',
                               label_predicted: str = 'Predicted') -> None:
    """
    Plots a number of points back from the last time series point as a zoomed view on observed
    and predicted data.
    :param observed: observed data
    :param predicted: predicted data
    :param num_points: number of points to draw on the horizontal axis
    :param title: plot title
    :param label_observed: label for the observed data
    :param label_predicted: label for the predicted data
    :return: None
    """
    plt.figure(figsize=(20, 10))
    plt.plot(observed.iloc[-num_points:], label=label_observed, c='orange')
    plt.plot(predicted[-num_points:], label=label_predicted, c='green')
    plt.title(title)
    plt.legend(loc='best')
    plt.show()


def plot_stl(data: pd.Series, period: int = 365, low_pass: int = 367) -> DecomposeResult:
    # TODO: dac opis

    # Seasonal-Trend decomposition - LOESS (STL)
    # from statsmodels.tsa.seasonal import STL
    # update with what this function returns
    # https://robjhyndman.com/hyndsight/seasonal-periods/

    stl = STL(data, period=period, low_pass=low_pass)
    result = stl.fit()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 16))
    ax1.set_title('Observed', loc='left')
    result.observed.plot(ax=ax1)
    ax2.set_title('Trend', loc='left')
    result.trend.plot(ax=ax2)
    ax3.set_title('Seasonal', loc='left')
    result.seasonal.plot(ax=ax3)
    ax4.set_title('Residuals', loc='left')
    result.resid.plot(ax=ax4)
    plt.show()

    return result


def plot_decompose(data: pd.Series):
    # update with what this function returns

    # TODO: dac opis

    result = seasonal_decompose(data, model='additive', freq=60)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 16))
    result.observed.plot(ax=ax1)
    result.trend.plot(ax=ax2)
    result.seasonal.plot(ax=ax3)
    result.resid.plot(ax=ax4)
    plt.show()

    return result


def plot_before_after(data_before: pd.Series,
                      data_after: pd.Series,
                      label_before: str = "Before",
                      label_after: str = "After") -> None:
    """
    Plots two charts for comparison.
    :param data_before: pandas Series
    :param data_after: pandas Series
    :param label_before: title for the first plot
    :param label_after: title for the second plot
    :return: None
    """

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
    ax1.set_title(label_before, loc='left')
    data_before.plot(ax=ax1)
    ax2.set_title(label_after, loc='left')
    data_after.plot(ax=ax2)
    plt.show()


#TODO: opisac
def plot_ts_corr(xt: pd.Series, nlag: int =30, fig_size=(20, 10)) -> None:
    # Function to plot signal, ACF and PACF
    if not isinstance(xt, pd.Series):
        xt = pd.Series(xt)
    plt.figure(figsize=fig_size)
    layout = (2, 2)

    # Assign axes
    ax_xt = plt.subplot2grid(layout, (0, 0), colspan=2)
    ax_acf = plt.subplot2grid(layout, (1, 0))
    ax_pacf = plt.subplot2grid(layout, (1, 1))

    # Plot graphs
    xt.plot(ax=ax_xt)
    ax_xt.set_title('Time Series')
    plot_acf(xt, lags=50, ax=ax_acf)
    plot_pacf(xt, lags=50, ax=ax_pacf)
    plt.tight_layout()
    return None
