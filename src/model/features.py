from datetime import datetime
import pandas as pd


def calculate_season(x: int) -> int:
    """
    Returns season number (winter is 1) based on a month number.
    :param x: month number, January is 1
    :return: None
    """
    seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
    # dec - feb is winter = 1, then spring, summer, fall etc.
    return seasons[x - 1]


def build_datetime_features(data: pd.DataFrame, dt_col_name: str) -> pd.DataFrame:
    """
    Removes Datetime index and calculates date-related features from it.
    :param data: pandas data frame
    :param dt_col_name: datetime column name
    :return: pandas data frame
    """
    # Move index to a column
    df = data.reset_index()

    # Build basic time features
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.html
    #df['year'] = pd.DatetimeIndex(df[dt_col_name]).year # year is jus like an identifier
    df['month'] = pd.DatetimeIndex(df[dt_col_name]).month
    df['day'] = pd.DatetimeIndex(df[dt_col_name]).day
    df['hour'] = pd.DatetimeIndex(df[dt_col_name]).hour
    df['dayofyear'] = pd.DatetimeIndex(df[dt_col_name]).dayofyear
    df['weekofyear'] = pd.DatetimeIndex(df[dt_col_name]).weekofyear
    df['dayofweek'] = pd.DatetimeIndex(df[dt_col_name]).dayofweek
    df['quarter'] = pd.DatetimeIndex(df[dt_col_name]).quarter

    # Seasons
    df['season'] = df['month'].map(calculate_season)

    df.drop([dt_col_name], axis=1, inplace=True)

    return df
