from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

def get_models_for_regression() -> list:
    """
    Defines a list of regression models to be used in the modelling.
    :return: list of base sklearn models
    """
    models = [LinearRegression(),
              ElasticNet(),
              SVR(gamma='scale'),
              DecisionTreeRegressor(),
              KNeighborsRegressor(),
              AdaBoostRegressor(),
              BaggingRegressor(n_estimators=10),
              RandomForestRegressor(n_estimators=10),
              ExtraTreesRegressor(n_estimators=10)
              ]
    return models