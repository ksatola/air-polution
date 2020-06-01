import pandas as pd
from sklearn.model_selection import train_test_split
from logger import logger
from utils import read_json


def load_data(data_file_path: str) -> None:
    """
    Loads a data set from path and displays shape and head().
    :param data_file_path: full path do data file
    :return: None
    """
    df = pd.read_csv(data_file_path, encoding='utf-8', sep=",", index_col="Datetime")
    logger.info(f'Dataframe loaded: {data_file_path}')
    logger.info(f'DataFrame size: {df.shape}')
    return df


def get_pm25_data_for_modelling(model_type: str = 'ml',
                                forecast_type: str = 'h') -> pd.DataFrame:
    """
    Reads HDF file with analytical view prepared for time series or machine learning modelling.
    :param model_type: 'ts' for time series analytical model or 'ml' for machine learning model
    :param forecast_type: 'd' for daily or 'h' for hourly data
    :return: pandas DataFrame
    """
    config = read_json('../config/pm25_model.json')
    data_folder = config['data_folder']

    if model_type == 'ml':
        if forecast_type == 'h':
            data_file_hdf = data_folder + 'dfpm25_2008-2018_ml_24hours_lags.hdf'
        else:
            data_file_hdf = data_folder + 'dfpm25_2008-2018_ml_7days_lags.hdf'
    else:
        if forecast_type == 'h':
            data_file_hdf = data_folder + 'dfpm25_2008-2018_hourly.hdf'
        else:
            data_file_hdf = data_folder + 'dfpm25_2008-2018_daily.hdf'

    df = pd.read_hdf(path_or_buf=data_file_hdf, key="df")
    logger.info(f'Dataframe loaded: {data_file_hdf}')
    logger.info(f'Dataframe size: {df.shape}')

    return df


def split_df_for_ml_modelling(data: pd.DataFrame, target_col: str = 't', train_size: float = 0.8) \
        -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Splits pandas DataFrame (columns as features, rows as observations) into train/test split 
    data frames deparately for independent and dependent features. 
    :param data: pandas DataFrame
    :param target_col: name of the target column
    :param train_size: train/test split ratio, 0-1, specifies how much data should be but in the
    train data set
    :return: tuple of four pandas DataFrames: X_train, X_test, y_train, y_test
    """
    # Split dataset into independent variables dataset columns and dependent variable column
    # X = df.iloc[:, 1:]
    # y = df.iloc[:, :1]
    X = data.copy()
    y = X.pop(target_col)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=train_size,
                                                        random_state=123)
    return X_train, X_test, y_train, y_test


def split_df_for_ts_modelling_percentage(data: pd.DataFrame,
                                         train_size: float = 0.8) -> (pd.DataFrame, pd.DataFrame):
    """
    Uses FIRST train_size of data for training / model parameters tuning and the later data for
    testing
    :param data: pandas DataFrame
    :param train_size: train/test split ratio, 0-1, specifies how much data should be but in the
    train data set
    :return: tuple of two pandas DataFrames: df_train and df_test
    """

    # Use FIRST train_size of data for training / model parameters tuning and the later data for
    # testing
    df_train = data[:int(len(data) * train_size)]
    df_test = data[int(len(data) * train_size):]
    # or
    # df_train = data.iloc[:-int(len(data) * 1-train_size)]
    # df_test = data.iloc[-int(len(data) * 1-train_size):]
    # or
    # X = data.values
    # train_size = int(len(X) * train_size)
    # df_train, df_test = X[0:train_size], X[train_size:len(X)]

    logger.info(f'Observations: {(len(data))}')
    logger.info(f'Training Observations: {(len(df_train))}')
    logger.info(f'Testing Observations: {(len(df_test))}')

    logger.info(f"{data.shape}, {df_train.shape}, {df_test.shape}, "
                f"{df_train.shape[0] + df_test.shape[0]}")

    return df_train, df_test


def split_df_for_ts_modelling_date_range(data: pd.DataFrame,
                                         train_range_from: str = '2008-01-01',
                                         train_range_to: str = '2016-12-31',
                                         test_range_from: str = '2017-01-01',
                                         test_range_to: str = None) -> (pd.DataFrame,
                                                                        pd.DataFrame):
    """
    Splits time series dataset based on dates as index
    :param data: pandas DataFrame with times series data (datetime format in the data frame's
    index)
    :param train_range_from: datetime string compliant with the dataset index format
    :param train_range_to: datetime string compliant with the dataset index format
    :param test_range_from: datetime string compliant with the dataset index format
    :param test_range_to: datetime string compliant with the dataset index format
    :return:
    """
    df = data.copy()
    df.index = pd.to_datetime(df.index)

    df_train = df[train_range_from:train_range_to].copy()
    df_test = df[test_range_from:test_range_to].copy()

    logger.info(f'Observations: {(len(data))}')
    logger.info(f'Training Observations: {(len(df_train))}')
    logger.info(f'Testing Observations: {(len(df_test))}')

    logger.info(f"{data.shape}, {df_train.shape}, {df_test.shape}, "
                f"{df_train.shape[0] + df_test.shape[0]}")

    return df_train, df_test
