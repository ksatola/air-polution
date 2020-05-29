import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from logger import logger


def load_data(data_file_path: str) -> None:
    """
    Loads a data set from path and displays shape and head().
    :param data_file_path:
    :return: None
    """
    df = pd.read_csv(data_file_path, encoding='utf-8', sep=",", index_col="Datetime")
    logger.info(f'DataFrame size: {df.shape}')
    display(df.head())
    return df


def get_ensemble_models_for_regression() -> list:
    """
    Defines a list of regression models to be used in ensemble modelling.
    :return: list of model
    """
    models = [LinearRegression(),
              ElasticNet(),
              SVR(gamma='scale'),
              DecisionTreeRegressor(),
              KNeighborsRegressor(),
              AdaBoostRegressor(),
              BaggingRegressor(n_estimators=10),
              RandomForestRegressor(n_estimators=10),
              ExtraTreesRegressor(n_estimators=10)]
    return models
