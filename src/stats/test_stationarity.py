from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import pandas as pd


def adfuller_test(data: pd.Series, alpha: float = 0.5) -> None:
    # TODO: dac opis
    result = adfuller(data)
    print(f"Test statistic: {result[0]}")
    print(f"P-value: {result[1]} -> {result[1]:.16f}")
    print(f"Critical values: {result[4]}")
    # Interpret
    p = result[1]
    if p > alpha:
        print('The time series has a unit root, so it is non-stationary. (fail to reject H0)')
    else:
        print('The time series does not have a unit root, so it is stationary (reject H0)')


# TODO: opisac i zrobic Interpret
def run_kpss_test(df: pd.DataFrame) -> None:
    print(" > Is the data stationary ?")
    # dftest = kpss(np.log(train['pm25']), 'ct')
    dftest = kpss(np.log(df), regression='c', nlags='auto')
    print("Test statistic = {:.3f}".format(dftest[0]))
    print("P-value = {:.3f}".format(dftest[1]))
    print("Critical values :")
    for k, v in dftest[3].items():
        print("\t{}: {}".format(k, v))
