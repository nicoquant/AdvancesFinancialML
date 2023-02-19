import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def getWeights(d, k):
    res = [1.]
    w = 1
    for i in range(1,k):
        res.insert(0, -w * (d-i+1) / i)
        w = -w * (d- i + 1) / i
    return np.array(res).reshape(k, 1)


def getWeights_FFD(d, thres):
    res, k = [1.], 1
    w = 1
    while True:
        w = -w * (d-k+1) / k
        if abs(w) < thres:
            break
        res.append(w)
        k += 1
    return np.array(res[::-1]).reshape(-1, 1)


def fracDiff(serie, d, threshold = 0.01):
    # compute weights for the longest serie
    w = getWeights(d, serie.shape[0])

    # initial calcs to be skipped based on the weight loss threshold
    w_ = np.cumsum(abs(w))
    w_ = w_ / w[-1]
    skip = w_[w_ > threshold].shape[0]  # skipping value lower than threshold

    # weights to values
    df = pd.Series(dtype="float64")
    serieF = serie.fillna(method='ffill').dropna()

    for iloc in range(skip, serieF.shape[0]):
        # select index: select series index
        # we take the iloc last values of the weight array
        loc = serieF.index[iloc]
        # weight * series
        df[loc] = (w[-(iloc+1):,:].T @ serieF.loc[:loc])[0]
    return df


def fracDiff_FFD(serie, d, threshold=0.01):
    """

    :param serie:
    :param d:
    :param threshold:
    :return: series differenciate by d in [0,1] with a fixed window to compute the weights
    """
    # compute weights
    w = getWeights_FFD(d, threshold)
    # nb of rows in returns series
    width = len(w) - 1

    df = pd.Series(dtype="float64")
    serieF = serie.fillna(method='ffill').dropna()

    for iloc in range(width, serieF.shape[0]):
        # define range of index: select series index
        loc0 = serieF.index[iloc-width]
        loc1 = serieF.index[iloc]
        # weight * serie
        df[loc1] = (w.T @ serieF.loc[loc0:loc1])[0]
    return df


def find_optimal_d(serie, step=0.1, choice_lag=1):
    stepping = int(1/step + 1)
    corr = {}
    dickey_fuller = {}
    d_optimal = "Non optimal d"
    ffd = "non optimal diff serie"
    minimum = True
    for d_opt in np.linspace(0, 1, stepping):
        # compute serie with a test d
        frac = fracDiff_FFD(serie, d_opt)
        # compute correlation
        corr["corr_series_" + str(round(d_opt,2))] = np.corrcoef(serie.loc[frac.index], frac)[0, 1]
        # compute df test of non stationarity

        # choice of max_lag = 1 has a huge impact on the computation of the df test.
        # Determining the lag using AIC leads to extremely different results
        dickey_fuller["p_value_5%_" + str(round(d_opt,2))] = adfuller(frac, regression='c', autolag=None, maxlag=choice_lag)[1]
        if dickey_fuller["p_value_5%_" + str(round(d_opt, 2))] <= 0.05 and minimum:
            d_optimal = d_opt.copy()
            ffd = frac
            minimum = False

    return corr, dickey_fuller, d_optimal, ffd

