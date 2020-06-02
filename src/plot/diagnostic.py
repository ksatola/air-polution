import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns


def plot_observations_to_predictions_relationship(observed: pd.Series,
                                                  predicted: pd.Series,
                                                  title: str = 'Observed vs. Predicted Percentage Change',
                                                  label_observed: str = 'Observations',
                                                  label_predicted: str = 'Predictions') -> None:
    """
    Plots a scatter plot and a fitted line of the predicted vs the observed percentage change.
    :param observed: observed data
    :param predicted: predicted data
    :param title: plot title
    :param label_observed: label for the observed data
    :param label_predicted: label for the predicted data
    :return: None
    """
    plt.figure(figsize=(20, 10))
    ax = sns.regplot(observed.pct_change(), predicted.pct_change())
    plt.xlabel(label_observed)
    plt.ylabel(label_predicted)
    plt.title(title)
    ax.grid(True, which='both')
    ax.axhline(y=0, color='#888888')
    ax.axvline(x=0, color='#888888')
    sns.despine(ax=ax, offset=0)
    # plt.xlim(-0.05, 0.05)
    # plt.ylim(-0.05, 0.05)
    plt.show()

def fit_theoretical_dist_and_plot(data: pd.Series) -> None:
    """
    Fits series of observations histogram into a normal probability density function (PDF).
    :param data: series of observations
    :return: None
    """
    # Fit a normal distribution to the data
    mu, std = norm.fit(data)

    plt.figure(figsize=(15, 10))

    # Plot the histogram
    plt.hist(data, bins=50, density=True, alpha=1.0)

    # Plot the PDF
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2, color='coral')
    plt.title(f"Fit results: mu = {mu:.2f},  std = {std:.2f}")
    plt.show()