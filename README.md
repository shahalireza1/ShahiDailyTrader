# Shahi Daily Trader

A lightweight daily-bar backtester with a built-in SMA crossover strategy, real historical data (via yfinance), cached downloads, and PNG visualizations.

## Installation

1. Ensure Python 3.9+ is installed.
2. (Recommended) Create and activate a virtual environment:
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux:**
     ```bash
     python -m venv .venv
     source .venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the backtester end-to-end with a single command. Example:

```bash
python -m trader backtest --symbol SPY --start 2018-01-01 --end 2025-01-01 --strategy sma_cross --fast 20 --slow 50 --starting-cash 100000 --position-fraction 1.0 --fee-bps 1.0 --slippage-bps 1.0
```

### What it does
- Downloads daily OHLCV data through **yfinance** (using adjusted close when available).
- Caches downloads to `./data_cache/` as CSV to avoid re-fetching.
- Applies the SMA crossover strategy (configurable `--fast` and `--slow`).
- Runs a long-only daily backtest with position sizing, fees (bps), and slippage (bps).
- Saves outputs under `./outputs/<timestamp>/`:
  - `signals_and_prices.csv` (data + signals)
  - `equity_curve.csv` (portfolio value over time)
  - `trades.csv` (entry/exit log)
  - `metrics.json` (CAGR, Sharpe, max drawdown, win rate, number of trades)
  - `price_signals.png` (price plus SMA and buy/sell markers)
  - `equity_curve.png` (equity curve)

### Quick smoke test
To validate everything without a long download, run a short window:
```bash
python -m trader backtest --symbol SPY --start 2023-01-01 --end 2024-01-01 --strategy sma_cross --fast 10 --slow 30
```

## Troubleshooting
- If you see a data error, double-check the symbol and date range.
- Delete files under `data_cache/` to force a fresh download if cached data looks stale.
- Use `--position-fraction` between 0 and 1 to control allocation; values outside the range are clipped.
