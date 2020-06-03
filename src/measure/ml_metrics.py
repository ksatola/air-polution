from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import *

import pandas as pd


def score_ml_models(X_train: pd.DataFrame,
                    y_train: pd.DataFrame,
                    models: list,
                    n_splits: int,
                    metric: str,
                    metric_label: str,
                    seed: int = 123) -> (list, list, list):
    """
    Calculates a metric for a dataset and list of models.
    :param X_train: pandas DataFrame with independent variables
    :param y_train: pandas DataFrame with dependent variable
    :param models: list of models
    :param n_splits: number of splits for kFold cross validation
    :param metric: scoring name as defined in
    https://scikit-learn.org/stable/modules/model_evaluation.html
    :param metric_label: label for the scoring name
    :param seed: random seed
    :return: tuple of three lists: output strings, cross validation results for each model,
    model name
    """
    names = []
    results = []
    output = []

    for name, model in models:
        names.append(name)
        try:
        # Not all scoring metrics are available for all models
            kfold = KFold(n_splits=n_splits, random_state=seed)
            cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=metric)
            results.append(cv_results)
            output.append(
            f'{name}, {metric_label} {-cv_results.mean()}, (std. dev. {cv_results.std()})')
        except:
            output.append(f'{name} {metric_label} metric unavailable)')

    return output, results, names
