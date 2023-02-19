from modules.fractionally_differentiated_features import *
import yfinance as yf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = yf.download('TSLA', start="2016-01-01", end="2023-02-19")
    # f = fracDiff(data['Close'], 1)
    # ff = fracDiff_FFD(data['Close'], 1)
    correl, df, d_opt, ffd = find_optimal_d(data['Close'])
    plt.figure(1)
    plt.plot(correl.values())
    plt.xlabel("Weight of differentiation")
    plt.ylabel("Correlation between Xt and Xt - w*Xt-1")
    plt.title("Evolution of correlation")
    plt.show()

    plt.figure(2)
    plt.plot(ffd)
    plt.show()




