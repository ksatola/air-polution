from statsmodels.tsa.stattools import adfuller


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

