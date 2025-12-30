# Shahi Daily Trader

A modular quantitative research platform that backtests portfolios across pluggable strategies, risk overlays, and analytics with YAML-driven configuration.

## Architecture

```
trader/
  core/        engine, portfolio accounting, execution placeholders, risk
  data/        loaders, cache helpers, external providers (sentiment/ML hooks)
  strategies/  base class + registry + packaged strategies (SMA/RSI, momentum, breakout, mean reversion)
  analytics/   metrics, plotting, reporting
  signals/     indicator utilities (SMA, RSI, z-score)
  utils/       configuration loader
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration (config.yaml)

```yaml
symbols:
  - SPY
  - QQQ
start: "2022-01-01"
end: "2023-12-31"
mode: backtest
fees_bps: 1.0
slippage_bps: 1.0
starting_cash: 100000
strategy:
  name: sma_rsi
  parameters:
    fast: 20
    slow: 50
    rsi_period: 14
    rsi_threshold: 55
risk:
  max_drawdown: 0.25
  target_volatility: 0.15
  vol_lookback: 20
  position_mode: fixed_fraction
  position_fraction: 0.8
  kelly_safety: 0.5
walkforward:
  train_window: 252
  test_window: 90
  step: 90
output_dir: outputs
```

## CLI

List all registered strategies:
```bash
python -m trader.backtest --list-strategies
```

Run a backtest from config:
```bash
python -m trader.cli --config config.yaml
```

Run walk-forward or paper modes:
```bash
python -m trader.cli --config config.yaml --run walkforward
python -m trader.cli --config config.yaml --run paper
```

Generate an HTML performance report (creates `outputs/<timestamp>/report.html`):
```bash
python -m trader.cli report --config configs/example.yaml
```

## Included Strategies
- **sma_cross**: traditional fast/slow crossover
- **sma_rsi**: crossover with RSI momentum filter
- **mean_reversion**: z-score reversion entry/exit
- **momentum**: lookback return breakout
- **breakout**: Donchian-style high/low breakout

## Risk & Portfolio Features
- Volatility targeting, Kelly-lite or fixed position sizing
- Max drawdown stop overlay
- Portfolio-level aggregation across multiple symbols

## Outputs (saved under `outputs/<timestamp>/`)
- `report.html` with equity vs buy-and-hold, drawdown, monthly returns, and metrics
- `equity_curve.csv` and PNG plots (equity, per-symbol cumulative returns, price with signals)
- `trades.csv` trade log
- `metrics.json` with CAGR, Sharpe, Sortino, max drawdown, hit rate, expectancy, profit factor, exposure
- Per-symbol signal/price CSVs

## Extensibility Hooks
- `core/execution.py`: broker adapters (paper + live placeholder)
- `data/providers.py`: sentiment/news + ML signal stubs ready for integration

## Tests

```bash
pytest
```
