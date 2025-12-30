from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity(equity_curve: pd.Series, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_curve.index, equity_curve.values, color="teal", label="Equity")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "equity_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_price_with_signals(df: pd.DataFrame, output_dir: Path, symbol: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df["Close"], color="black", label=f"{symbol} Close")
    if "fast_sma" in df.columns:
        ax.plot(df.index, df["fast_sma"], label="Fast SMA", alpha=0.7)
    if "slow_sma" in df.columns:
        ax.plot(df.index, df["slow_sma"], label="Slow SMA", alpha=0.7)
    signals = df[df["signal"].diff().fillna(df["signal"]).abs() > 0]
    ax.scatter(signals.index, signals["Close"], color="green", marker="o", s=20, label="Signal")
    ax.legend()
    ax.set_title(f"{symbol} Prices & Signals")
    fig.tight_layout()
    path = output_dir / f"{symbol}_signals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_symbol_returns(per_symbol: Dict[str, pd.Series], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    for sym, series in per_symbol.items():
        ax.plot(series.index, series.cumsum(), label=sym)
    ax.set_title("Cumulative Returns by Symbol")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "per_symbol_returns.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
