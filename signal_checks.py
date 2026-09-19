import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from residual_signal import get_factor_returns

def check_distribution(signals):
    s = signals.stack().dropna()

    print('count:', len(s))
    print('mean:', round(s.mean(), 3))
    print('std:', round(s.std(),3))
    print('% |s| > 1.25:', round((s.abs() > 1.25).mean() * 100, 2))
    print('% |s| > 2:', round((s.abs() > 2).mean() * 100, 2))

    plt.hist(s.clip(-5, 5), bins=100)
    plt.axvline(-2, color='red')
    plt.axvline(2, color='red')
    plt.title('s-score distribution')
    plt.show()

def check_pass_rate(returns, signals, window=60):
    nan_count = returns.isna().rolling(window).sum().sum(axis=1)
    ok_day = (nan_count == 0) & (np.arange(len(returns)) >= window - 1)

    rate = signals.notna().sum(axis=1) / signals.shape[1]
    rate = rate[ok_day]

    print('average pass rate:', round(rate.mean(), 3))
    print('min:', round(rate.min(), 3), 'max:', round(rate.max(), 3))

    rate.plot(title='fraction of stocks passing filters')
    plt.show()

def check_ticker(returns, ticker,t, window=60, k=1):
    j = returns.columns.get_loc(ticker)
    r = returns.iloc[t - window + 1: t+1].values
    f = get_factor_returns(returns, t, window, k)

    x = np.column_stack([np.ones(window), f])
    coef = np.linalg.lstsq(x, r[:, j], rcond=None)[0]
    resid = r[:, j] - x @ coef
    cum = np.cumsum(resid)

    x_ct = cum[:-1]
    y_ct = cum[1:]
    b, a = np.polyfit(x_ct, y_ct, 1)

    if b <= 0 or b >= 1:
        print(ticker, "-> b =", round(b, 3), 'not mean reverting')
        return

    kappa = -np.log(b) * 252
    m = a / (1-b)
    errors = y_ct - (a+b * x_ct)
    sigma_eq = np.sqrt(errors.var(ddof=1)/(1-b ** 2))
    s = (cum[-1] - m)/sigma_eq

    print(ticker, 'b:', round(b,3), 'kappa:', round(kappa, 1), 's-score:', round(s, 2))
    plt.plot(returns.index[t - window + 1 : t+1], cum, label = 'cum residual')
    plt.axhline(m, color='black', label='m')
    plt.axhline(m + sigma_eq, color='red', linestyle='--')
    plt.axhline(m - sigma_eq, color='blue', linestyle='--')
    plt.title(ticker + 's-score=' + str(round(s, 2)))
    plt.legend()
    plt.show()

