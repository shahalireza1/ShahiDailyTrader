from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
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


def plot_equity_with_drawdown(
    equity_curve: pd.Series, benchmark_curve: pd.Series, output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, height_ratios=[3, 1.4])
    ax_top.plot(equity_curve.index, equity_curve.values, color="teal", label="Strategy")
    ax_top.plot(
        benchmark_curve.index,
        benchmark_curve.values,
        color="gray",
        linestyle="--",
        label="Buy & Hold",
    )
    ax_top.set_title("Equity Curve vs. Buy & Hold")
    ax_top.set_ylabel("Portfolio Value")
    ax_top.legend()

    ax_bottom.fill_between(drawdown.index, drawdown.values, color="salmon")
    ax_bottom.set_title("Drawdown")
    ax_bottom.set_ylabel("Drawdown")
    ax_bottom.set_xlabel("Date")
    ax_bottom.set_ylim(drawdown.min() * 1.05 if not drawdown.empty else -0.1, 0)

    fig.tight_layout()
    path = output_dir / "equity_and_drawdown.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_equity_vs_benchmark(
    equity_curve: pd.Series, benchmark_curve: pd.Series, output_dir: Path
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity_curve.index, equity_curve.values, color="teal", label="Strategy")
    ax.plot(benchmark_curve.index, benchmark_curve.values, color="gray", linestyle="--", label="Buy & Hold")
    ax.set_title("Equity vs. Buy & Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "equity_vs_benchmark.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_drawdown(equity_curve: pd.Series, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(drawdown.index, drawdown.values, color="salmon")
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylim(drawdown.min() * 1.05 if not drawdown.empty else -0.1, 0)
    fig.tight_layout()
    path = output_dir / "drawdown.png"
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


def plot_monthly_returns_heatmap(monthly_pivot: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    working = monthly_pivot.copy()
    if working.empty:
        working = pd.DataFrame([[0.0]], index=["N/A"], columns=["N/A"])
    max_abs = working.abs().max().max()
    max_abs = max_abs if max_abs > 0 else 1

    fig, ax = plt.subplots(figsize=(10, 4))
    cax = ax.imshow(working.values, cmap="RdYlGn", aspect="auto", vmin=-max_abs, vmax=max_abs)
    ax.set_xticks(range(len(working.columns)))
    ax.set_xticklabels(working.columns)
    ax.set_yticks(range(len(working.index)))
    ax.set_yticklabels(working.index)
    ax.set_title("Monthly Returns (%)")
    fig.colorbar(cax, ax=ax, orientation="vertical", label="Return %")
    fig.tight_layout()
    path = output_dir / "monthly_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
