import pandas as pd
from sklearn.model_selection import train_test_split
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


def get_data_for_modelling(model_type: list = ['ts', 'ml']) -> (
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame):
    read_json(args.config_json)



def split_df_for_ml_modelling(data: pd.DataFrame, target_col: str, test_size: float) -> (
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame):
    """
    Splits pandas DataFrame (columns as features, rows as observations) into train/test split 
    data frames deparately for independent and dependent features. 
    :param data: pandas DataFrame
    :param target_col: name of the target column
    :param test_size: train/test split ratio, 0-1, specifies how much data should be but in the
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
                                                        test_size=0.5,
                                                        random_state=123)
    return X_train, X_test, y_train, y_test
