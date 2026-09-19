"""
residual_signals.py

"""
import numpy as np
import pandas as pd

def get_factor_returns(returns, t, window=60, k=1):
    r = returns.iloc[t-window+1:t+1].values

    mean = r.mean(axis=0)
    std = r.std(axis=0)
    z = (r-mean) /std

    corr = np.corrcoef(z, rowvar=False)
    eigenvals, eigenvecs = np.linalg.eigh(corr)

    eigenvals = eigenvals[::-1]
    eigenvecs =eigenvecs[:, ::-1]

    v = eigenvecs[:, :k].copy()

    for i in range(k):
        if v[:, i].sum()<0:
            v[:, i] = -v[:, i]

    weights = v / std.reshape(-1, 1)

    f = r @ weights
    return f 

def get_sscores(r, f):
    w, n = r.shape

    x = np.column_stack([np.ones(w), f])

    s_scores = np.full(n, np.nan)

    for j in range(n):
        stock = r[:, j]

        coef = np.linalg.lstsq(x, stock, rcond=None)[0]
        resid = stock - x @ coef

        cum = np.cumsum(resid)

        x_ar = cum[:-1]
        y_ar = cum[1:]
        b, a = np.polyfit(x_ar, y_ar, 1)

        if b <= 0 or b >= 1:
            continue 

        kappa = -np.log(b) * 252
        if kappa < 252 / 30:
            continue

        m = a / (1-b)
        errors = y_ar - (a+b*x_ar)
        sigma_eq = np.sqrt(errors.var(ddof=1)/(1-b ** 2))

        s_scores[j] = (cum[-1] - m) / sigma_eq
        s_scores = s_scores - np.nanmean(s_scores)
        
    return s_scores

def build_signals(returns, window=60, k=1):
    signals = pd.DataFrame(np.nan, index=returns.index, columns=returns.columns)

    for t in range(window -1, len(returns)):
        r = returns.iloc[t-window + 1 :  t+1].values

        if np.isnan(r).any():
            continue

        f = get_factor_returns(returns, t, window, k)
        signals.iloc[t] = get_sscores(r, f)

    return signals 