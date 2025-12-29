from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    output_dir: Path


def _calculate_metrics(equity_curve: pd.Series, trades: pd.DataFrame, starting_cash: float) -> Dict[str, float]:
    if equity_curve.empty:
        return {
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "num_trades": 0,
        }

    daily_returns = equity_curve.pct_change().dropna()
    total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    annual_factor = 365 / total_days if total_days else 0
    cagr = (equity_curve.iloc[-1] / starting_cash) ** annual_factor - 1 if annual_factor else 0.0

    sharpe = 0.0
    if not daily_returns.empty and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min() if not drawdown.empty else 0.0

    win_rate = 0.0
    num_trades = len(trades)
    if num_trades > 0:
        win_rate = (trades["pnl"] > 0).mean()

    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "num_trades": int(num_trades),
    }


def _generate_trades(df: pd.DataFrame) -> pd.DataFrame:
    trades: List[Dict[str, object]] = []
    position_open_date: Optional[pd.Timestamp] = None
    position_open_price: Optional[float] = None

    prev_signal = 0
    for ts, row in df.iterrows():
        signal = int(row["signal"])
        price = float(row["Close"])

        if prev_signal == 0 and signal == 1:
            position_open_date = ts
            position_open_price = price
        elif prev_signal == 1 and signal == 0 and position_open_price is not None:
            pnl = (price - position_open_price) / position_open_price
            trades.append(
                {
                    "entry_date": position_open_date,
                    "exit_date": ts,
                    "entry_price": position_open_price,
                    "exit_price": price,
                    "pnl": pnl,
                }
            )
            position_open_date = None
            position_open_price = None

        prev_signal = signal

    # Close any open position at the final price
    if prev_signal == 1 and position_open_price is not None:
        last_price = float(df["Close"].iloc[-1])
        last_ts = df.index[-1]
        pnl = (last_price - position_open_price) / position_open_price
        trades.append(
            {
                "entry_date": position_open_date,
                "exit_date": last_ts,
                "entry_price": position_open_price,
                "exit_price": last_price,
                "pnl": pnl,
            }
        )

    return pd.DataFrame(trades)


def run_backtest(
    df: pd.DataFrame,
    starting_cash: float = 100_000,
    position_fraction: float = 1.0,
    fee_bps: float = 1.0,
    slippage_bps: float = 1.0,
    output_dir: Optional[Path] = None,
) -> BacktestResult:
    """Run a long-only daily backtest using signal-based position sizing."""

    if "signal" not in df.columns:
        raise ValueError("DataFrame must contain a 'signal' column. Did you run a strategy?")

    working = df.copy()
    working.sort_index(inplace=True)
    working["return"] = working["Close"].pct_change().fillna(0)

    position_fraction = max(0.0, min(1.0, position_fraction))
    position = working["signal"].shift(1).fillna(0) * position_fraction

    transaction_rate = (fee_bps + slippage_bps) / 10_000
    position_change = position.diff().fillna(position)
    transaction_cost = abs(position_change) * transaction_rate

    strategy_return = position * working["return"] - transaction_cost
    equity_curve = (1 + strategy_return).cumprod() * starting_cash
    equity_curve.name = "equity"

    trades = _generate_trades(working)
    metrics = _calculate_metrics(equity_curve, trades, starting_cash)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        equity_curve.to_csv(output_dir / "equity_curve.csv", header=["equity"])
        working.to_csv(output_dir / "signals_and_prices.csv")
        trades.to_csv(output_dir / "trades.csv", index=False)

    return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=metrics, output_dir=output_dir or Path("."))
