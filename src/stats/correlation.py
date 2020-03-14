import pandas as pd

def pearson_2d_corr(df: pd.DataFrame, col1: str, col2: str) -> float:
    #TODO docstring + test for more columns
    corr = df[col1].corr(df[col2])
    return corr