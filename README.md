# PCA Statistical Arbitrage

A statistical arbitrage strategy using PCA-based eigenportfolios on Financial sector stocks.

## Idea

- Extract common risk factors from a basket of correlated stocks using PCA.
- Each stock's returns are regressed on these factors.
- What's left over is stock-specific movement, not explained by the market.
- Trade the residual when it looks stretched (mean-reversion bet).

## Universe

- Financial sector tickers
- Daily OHLCV data pulled via  ```yfinance```

## Pipeline

1. ### Data Loading (```data_loader.py```)
    - Fetch tickers, OHLCV data
    - Compute returns
    - Clean data, sanity check with correlation map

2. ### PCA / Eigenportfolio (```pca_factors.py```)
    - Standardize returns
    - Run PCA
    - Select `k=1` (PC1) as the dominant factor
    - Sanity check: PC1 weights are all +ve across 74 stocks

3. ### Residual Signal (```residual_signal.py```)
    - Regress each stock's returns on eigenporfolio
    - compute residuals
    - convert residuals into a z-score for signal generation

4. ### Backtest (```backtest.py```)
    - Test the residual-based mean-reversion strategy
    - evaluate performance

5. ### Main (```main.ipynb```)
    - runs the full pipeline end-to-end

## File Structure

pca-stat-arb/

|-- data_loader.py

|-- pca_factors.py

|-- residual_signal.py

|-- backtest.py

|-- main.ipynb

|-- README.md

## Status

- ✅ Phase 1: Data loading
- ✅ Phase 2: PCA / eigenportfolios
- 🔄 Phase 3: residual signal (in progress)
- ⬜️ Phase 4: Backtest

## Requirements

numpy
pandas
scikit-learn
yfinance
matplotlib 