"""This package is dedicated for time series related plots."""

from .ts_plots import (
    plot_train_test_predicted,
    plot_observed_vs_predicted,
    plot_stl
)

from .diagnostic import (
    plot_observations_to_predictions_relationship,
    fit_theoretical_dist_and_plot
)