from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator

from sklearn.model_selection import KFold

from logger import logger
import pandas as pd

from measure import get_rmse


def get_analytical_view_for_meta_model(X_train: pd.DataFrame,
                                       y_train: pd.DataFrame,
                                       models: list,
                                       n_splits: int = 10
                                       ) -> (pd.DataFrame, pd.DataFrame):
    """
    Builds an analytical view based on kFold predictions of base models to be used by another
    model (out-of-fold predictions).
    :param X_train: input variables (pandas DataFrame)
    :param y_train: target variables (pandas DataFrame)
    :param models: list of base models
    :param n_splits: number of KFold splits
    :return: tuple of two pandas DataFrames, one with new input variables, and second with
    corresponding target variables (a copy of y_train)
    """

    # Define split of data
    kfold = KFold(n_splits=n_splits, shuffle=True)

    # Prepare empty data frame with column names (columns as models)
    columns = [type(model).__name__ for model in models]
    # and add target column
    columns = columns.append(y_train.columns[0])
    meta_X = pd.DataFrame(columns=columns)

    cv_fold_number = 1

    for train_indices, test_indices in kfold.split(X_train):
        logger.debug(f'train: {train_indices}, len: {len(train_indices)}')
        logger.debug(f'test: {test_indices}, len: {len(test_indices)}')

        logger.info(f'CV fold number -> {cv_fold_number}')
        cv_fold_number += 1

        # Get data
        train_X, test_X = X_train.iloc[train_indices], X_train.iloc[test_indices]
        train_y, test_y = y_train.iloc[train_indices], y_train.iloc[test_indices]
        logger.debug(f'train_indices {train_indices.shape}')
        logger.debug(f'test_indices {test_indices.shape}')

        # Add target variable
        fold_yhats = test_y.copy()

        # Fit and make predictions with each sub-model
        for model in models:
            model_name = type(model).__name__
            model.fit(train_X, train_y)
            yhat = model.predict(test_X)
            logger.info(model_name)
            logger.debug(f'train_X.shape {train_X.shape}')
            logger.debug(f'train_y.shape {train_y.shape}')
            logger.debug(f'yhat.shape {yhat.shape}')

            # Build fold-level results data frame, models as features
            fold_yhats[f'{model_name}'] = yhat

        logger.debug(f'meta_X shape {meta_X.shape}')
        meta_X = pd.concat([meta_X, fold_yhats])

    # Take the target variable out from the dataset
    meta_y = meta_X.pop(y_train.columns[0]).to_frame()

    return meta_X, meta_y


def fit_base_models(X_train: pd.DataFrame,
                    y_train: pd.DataFrame,
                    models) -> list:
    """
    Fits all base models on the training dataset.
    :param X_train: input variables (pandas DataFrame)
    :param y_train: target variables (pandas DataFrame)
    :param models: list of base models
    :return: list of fitted sklearn models
    """
    for model in models:
        model.fit(X_train, y_train)

    return models


def fit_meta_model(X_train: pd.DataFrame,
                   y_train: pd.DataFrame,
                   meta_model: BaseEstimator = LinearRegression()) -> BaseEstimator:
    """
    Fits a meta model on the training dataset.
    :param X_train: input variables (pandas DataFrame)
    :param y_train: target variables (pandas DataFrame)
    :param meta_model: sklearn regression estimator class
    :return: fitted sklearn model
    """
    model = meta_model
    model.fit(X_train, y_train)
    return model


def evaluate_models(X_test: pd.DataFrame, y_test: pd.DataFrame, models: list) -> None:
    """
    Evaluates a list of models on a dataset using RMSE (Root Mean Squared Error).
    :param X_test: input variables (pandas DataFrame)
    :param y_test: target variables (pandas DataFrame)
    :param models: list of base models
    :return: None
    """
    for model in models:
        yhat = model.predict(X_test)
        logger.info(f'{model.__class__.__name__} RMSE {get_rmse(y_test, yhat)}')


def predict_with_super_learner(X_test: pd.DataFrame,
                               y_test: pd.DataFrame,
                               models: list,
                               meta_model: BaseEstimator) -> (pd.DataFrame, pd.DataFrame):
    """
    Makes predictions with stacked (meta) model.
    :param X_test: input variables (pandas DataFrame)
    :param y_test: target variables (pandas DataFrame)
    :param models: list of trained base models
    :param meta_model: sklearn model used for stacking
    :return: tuple of two pandas DataFrames, one with meta-model predictions, and second with
    corresponding target variables (a copy of y_test)
    """
    # yhats = y_test.copy()

    # Prepare empty data frame with column names (columns as models)
    columns = [type(model).__name__ for model in models]
    # and add target column
    columns = columns.append(y_test.columns[0])
    models_yhat = pd.DataFrame(columns=columns)

    # Add target variable
    yhats = y_test.copy()

    for model in models:
        model_name = type(model).__name__
        yhat = model.predict(X_test)
        logger.info(model_name)
        logger.debug(f'yhat.shape {yhat.shape}')

        # Build results data frame, models as features
        yhats[f'{model_name}'] = yhat

    models_yhat = pd.concat([models_yhat, yhats])
    # Take the target variable out from the dataset
    meta_target = models_yhat.pop(y_test.columns[0]).to_frame()

    # Predict
    meta_yhat = meta_model.predict(models_yhat)

    return meta_yhat, meta_target
