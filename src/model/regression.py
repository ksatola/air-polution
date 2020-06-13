from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

from logger import logger


def get_models_for_regression() -> list:
    """
    Defines a list of linear and non-linear regression models to be used in the modelling.
    :return: list of base sklearn models
    """
    models = [
        LinearRegression(),  # OLS
        ElasticNet(),
        SVR(),
        # SVR(kernel='linear', C=100, gamma='auto'),
        # SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1),
        DecisionTreeRegressor(),
        KNeighborsRegressor(),
        AdaBoostRegressor(),
        BaggingRegressor(n_estimators=10),
        RandomForestRegressor(n_estimators=10),
        ExtraTreesRegressor(n_estimators=10)
    ]
    return models


def perform_grid_search_cv(X_train,
                           y_train,
                           model,
                           param_grid,
                           scoring,
                           num_folds=6,
                           seed=123):
    """
    Performs cross-validation hyper-parameters tuning on k-folded dataset
    :param X_train: observations, independent variables
    :param y_train: observations, dependent variable
    :param model: BaseEstimator type of model (sklearn)
    :param param_grid: dictionary of hyper-parameters with their values
    :param scoring: sklearn scoring string
    :param num_folds: number of folds for cross-validation
    :param seed: random seed
    :return: None
    """
    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, y_train)

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    # for mean, stdev, param in zip(means, stds, params):
    # print("{:0.2f} ({:0.2f}) with: {}".format(mean, stdev, param))
    # print('-------')
    logger.info(f'Best: {grid_result.best_score_} using {grid_result.best_params_}')


def perform_random_search_cv(X_train,
                             y_train,
                             model,
                             param_grid,
                             scoring,
                             num_folds=6,
                             seed=123):
    """

    :param X_train:
    :param y_train:
    :param model:
    :param param_grid:
    :param scoring:
    :param num_folds:
    :param seed:
    :return:
    """
    kfold = KFold(n_splits=num_folds, random_state=seed)
    grid = RandomizedSearchCV(estimator=model,
                              param_distributions=param_grid,
                              scoring=scoring,
                              verbose=1,
                              n_jobs=-1,
                              n_iter=1000)
    grid_result = grid.fit(X_train, y_train)

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    # for mean, stdev, param in zip(means, stds, params):
    # print("{:0.2f} ({:0.2f}) with: {}".format(mean, stdev, param))
    # print('-------')
    logger.info(f'Best: {grid_result.best_score_} using {grid_result.best_params_}')
