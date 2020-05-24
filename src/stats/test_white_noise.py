
def white_noise_check(data: pd.Series) -> None:
    # TODO: dac opis
    X = data.values
    split = int(len(X) / 2)
    X1, X2 = X[0:split], X[split:]
    mean1, mean2 = X1.mean(), X2.mean()
    var1, var2 = X1.var(), X2.var()
    print('Mean1=%f, Mean2=%f' % (mean1, mean2))
    print('Variance1=%f, Variance2=%f' % (var1, var2))

    plt.figure(figsize=(20, 16))
    autocorrelation_plot(data)
    plt.show()