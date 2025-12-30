from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_annualized_return(equity_curve: pd.Series, starting_cash: float) -> float:
    if equity_curve.empty:
        return 0.0
    daily_returns = equity_curve.pct_change().dropna()
    trading_years = len(daily_returns) / 252 if len(daily_returns) else 0
    if trading_years == 0:
        return 0.0
    return (equity_curve.iloc[-1] / starting_cash) ** (1 / trading_years) - 1


def compute_metrics(equity_curve: pd.Series, trades: pd.DataFrame, starting_cash: float) -> dict:
    daily_returns = equity_curve.pct_change().dropna()
    total_return = (equity_curve.iloc[-1] - starting_cash) / starting_cash if not equity_curve.empty else 0.0
    cagr = _safe_annualized_return(equity_curve, starting_cash)

    sharpe = 0.0
    if not daily_returns.empty and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    downside = daily_returns[daily_returns < 0]
    sortino = 0.0
    if not downside.empty and downside.std() > 0:
        sortino = (daily_returns.mean() / downside.std()) * np.sqrt(252)

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min() if not drawdown.empty else 0.0

    win_rate = 0.0
    expectancy = 0.0
    num_trades = len(trades)
    if num_trades:
        wins = trades[trades["pnl"] > 0]
        losses = trades[trades["pnl"] <= 0]
        win_rate = len(wins) / num_trades
        avg_win = wins["pnl"].mean() if not wins.empty else 0.0
        avg_loss = losses["pnl"].mean() if not losses.empty else 0.0
        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "expectancy": float(expectancy),
        "num_trades": int(num_trades),
    }
