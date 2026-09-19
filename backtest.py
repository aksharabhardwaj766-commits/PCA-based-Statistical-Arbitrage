'''
backtest.py
'''
import numpy as np
import pandas as pd

def get_positions(signal, entry=1.25, exit_long=0.05, exit_short=0.75):
    # +1 long, -1 short, 0 no position

    pos = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    prev = pd.Series(0.0, index=signal.columns)

    for t in range(len(signal)):
        s = signal.iloc[t]
        cur = prev.copy()

        cur[(prev == 0) & (s < -entry)] = 1.0
        cur[(prev == 0) & (s > entry)] = -1.0

        cur[(prev == 1) & (s > -exit_long)] = 0.0
        cur[(prev == -1) & (s < exit_short)] = 0.0

        cur[s.isna()] = 0.0

        pos.iloc[t] = cur
        prev = cur

    return pos

def get_weights(pos, gross_per_side=0.5, n_max=30):
    per_name = gross_per_side / n_max
    w = pos * per_name

    longs = w.clip(lower=0)
    shorts = w.clip(upper=0)

    long_sum = longs.sum(axis=1)
    short_sum = -shorts.sum(axis=1)

    long_scale = (gross_per_side / long_sum).clip(upper=1).fillna(1)
    short_scale = (gross_per_side / short_sum).clip(upper=1).fillna(1)

    w = longs.mul(long_scale, axis=0) + shorts.mul(short_scale, axis=0)

    return w

def run_backtest(returns, signal, cost_bps=5):
    dates = returns.index.intersection(signal.index)
    tickers = returns.columns.intersection(signal.columns)
    returns = returns.loc[dates, tickers]
    signal = signal.loc[dates, tickers]

    pos = get_positions(signal)
    w = get_weights(pos)

    w = w.shift(1).fillna(0)

    gross = (w * returns).sum(axis=1)

    turnover = w.diff().abs().sum(axis=1).fillna(0)
    cost = turnover * cost_bps / 10000
    net = gross - cost 

    equity = (1 + net).cumprod()

    return w, gross, net, turnover, equity


def get_metrics(net, turnover):
    sharpe = np.sqrt(252) * net.mean() / net.std()

    equity = (1 + net).cumprod()
    dd = equity / equity.cummax()-1

    print('sharpe:', round(sharpe, 2))
    print('annual returns:', round(net.mean() * 252, 3))
    print('annual vol:', round(net.std() * np.sqrt(252), 3))
    print('max drawdown:', round(dd.min(), 3))
    print('annual turnover:', round(turnover.mean() * 252, 1))
    print('hit rate:', round((net > 0).mean(), 3))

    return sharpe

def cost_check(returns, signal):
    for c in [0, 5, 10]:
        w, gross, net, turnover, equity = run_backtest(returns, signal, cost_bps=c)
        sharpe = np.sqrt(252) * net.mean() / net.std()
        print(c, 'bps -> sharpe', round(sharpe, 2))