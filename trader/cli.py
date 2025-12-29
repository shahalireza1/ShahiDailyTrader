import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd

from trader.backtest import BacktestResult, run_backtest
from trader.data import fetch_price_data
from trader.strategy import sma_crossover


STRATEGIES = {
    "sma_cross": sma_crossover,
}


def _plot_price_and_signals(df: pd.DataFrame, trades: pd.DataFrame, output_dir: Path, symbol: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], label=f"{symbol} Close", color="black")
    ax.plot(df.index, df.get("fast_sma"), label="Fast SMA", color="blue", alpha=0.7)
    ax.plot(df.index, df.get("slow_sma"), label="Slow SMA", color="orange", alpha=0.7)

    if not trades.empty:
        ax.scatter(trades["entry_date"], trades["entry_price"], marker="^", color="green", label="Buy", zorder=5)
        ax.scatter(trades["exit_date"], trades["exit_price"], marker="v", color="red", label="Sell", zorder=5)

    ax.set_title(f"{symbol} Price with Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "price_signals.png", dpi=150)
    plt.close(fig)


def _plot_equity_curve(result: BacktestResult, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(result.equity_curve.index, result.equity_curve.values, label="Equity")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=150)
    plt.close(fig)


def _run_strategy(name: str, df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    strategy_func: Callable = STRATEGIES.get(name)
    if not strategy_func:
        raise ValueError(f"Unknown strategy '{name}'. Available: {', '.join(STRATEGIES)}")

    if name == "sma_cross":
        return strategy_func(df, fast=fast, slow=slow)

    return strategy_func(df)


def run_backtest_cli(args: argparse.Namespace) -> None:
    try:
        df = fetch_price_data(args.symbol, args.start, args.end)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load data: {exc}")
        return

    try:
        df = _run_strategy(args.strategy, df, fast=args.fast, slow=args.slow)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to compute strategy: {exc}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / timestamp
    try:
        result = run_backtest(
            df,
            starting_cash=args.starting_cash,
            position_fraction=args.position_fraction,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            output_dir=output_dir,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to run backtest: {exc}")
        return

    _plot_price_and_signals(df, result.trades, output_dir, args.symbol)
    _plot_equity_curve(result, output_dir)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result.metrics, indent=2))

    print("\nBacktest complete!")
    print(f"Outputs saved to: {output_dir.resolve()}")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trading backtester")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest_parser = subparsers.add_parser("backtest", help="Run a backtest over historical data")
    backtest_parser.add_argument("--symbol", required=True, help="Ticker symbol, e.g., SPY")
    backtest_parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    backtest_parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    backtest_parser.add_argument("--strategy", default="sma_cross", help="Strategy name")
    backtest_parser.add_argument("--fast", type=int, default=20, help="Fast SMA window")
    backtest_parser.add_argument("--slow", type=int, default=50, help="Slow SMA window")
    backtest_parser.add_argument("--starting-cash", dest="starting_cash", type=float, default=100_000, help="Starting portfolio value")
    backtest_parser.add_argument("--position-fraction", dest="position_fraction", type=float, default=1.0, help="Fraction of capital to deploy on signals (0-1)")
    backtest_parser.add_argument("--fee-bps", dest="fee_bps", type=float, default=1.0, help="Trading fee in basis points per trade")
    backtest_parser.add_argument("--slippage-bps", dest="slippage_bps", type=float, default=1.0, help="Slippage in basis points per trade")
    backtest_parser.set_defaults(func=run_backtest_cli)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
