import pandas as pd
import yfinance as yf
import requests 

def get_financial_tickers():

    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)

    tables = pd.read_html(response.text)
    sp500 = tables[0]

    financials = sp500[sp500['GICS Sector'] == 'Financials']
    tickers = financials['Symbol'].tolist() 

    tickers = [t.replace('.', '-') for t in tickers]

    return tickers

def get_returns(tickers, start='2018-01-01', end='2025-01-01'):
    prices = yf.download(tickers, start=start, end=end, auto_adjust=True)['Close']
    returns = prices.pct_change()
    returns = returns.dropna(axis=1, thresh=int(0.95 * len(returns)))
    returns = returns.dropna()
    print('after:', returns.shape)
    return returns

