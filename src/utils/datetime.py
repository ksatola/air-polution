from datetime import datetime


def get_datetime_identifier(dt_format: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """
    Returns a datetime formatted string of now() timestamp
    :param dt_format: format of date time output string
    :return: datetime string
    """
    return datetime.now().strftime(dt_format)
