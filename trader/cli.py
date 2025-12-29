import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd

from trader.backtest import BacktestResult, run_backtest
from trader.data import fetch_price_data
from trader.strategy import SMARsiConfig, sma_crossover, sma_rsi_filter


STRATEGIES = {
    "sma_cross": sma_crossover,
    "sma_rsi": sma_rsi_filter,
}


def _plot_results(df: pd.DataFrame, result: BacktestResult, output_dir: Path, symbol: str, show: bool) -> None:
    fig, (ax_price, ax_eq) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    ax_price.plot(df.index, df["Close"], label=f"{symbol} Close", color="black", linewidth=1.2)
    if "fast_sma" in df.columns:
        ax_price.plot(df.index, df["fast_sma"], label="Fast SMA", color="blue", alpha=0.8)
    if "slow_sma" in df.columns:
        ax_price.plot(df.index, df["slow_sma"], label="Slow SMA", color="orange", alpha=0.8)
    if "rsi" in df.columns:
        ax_rsi = ax_price.twinx()
        ax_rsi.plot(df.index, df["rsi"], label="RSI", color="purple", alpha=0.4)
        ax_rsi.axhline(50, color="purple", linestyle="--", alpha=0.3)
        ax_rsi.set_ylabel("RSI")

    trades = result.trades
    if not trades.empty:
        ax_price.scatter(trades["entry_date"], trades["entry_price"], marker="^", color="green", label="Buy", zorder=5)
        ax_price.scatter(trades["exit_date"], trades["exit_price"], marker="v", color="red", label="Sell", zorder=5)

    ax_price.set_title(f"{symbol} Price & Signals")
    ax_price.set_ylabel("Price")
    ax_price.legend(loc="upper left")

    ax_eq.plot(result.equity_curve.index, result.equity_curve.values, color="teal", label="Equity")
    ax_eq.set_title("Equity Curve")
    ax_eq.set_xlabel("Date")
    ax_eq.set_ylabel("Portfolio Value")
    ax_eq.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "summary.png", dpi=150)

    if show:
        plt.show()

    plt.close(fig)


def _run_strategy(name: str, df: pd.DataFrame, fast: int, slow: int, rsi_period: int, rsi_threshold: float) -> pd.DataFrame:
    strategy_func: Callable = STRATEGIES.get(name)
    if not strategy_func:
        raise ValueError(f"Unknown strategy '{name}'. Available: {', '.join(STRATEGIES)}")

    if name == "sma_cross":
        return strategy_func(df, fast=fast, slow=slow)

    if name == "sma_rsi":
        config = SMARsiConfig(fast=fast, slow=slow, rsi_period=rsi_period, rsi_threshold=rsi_threshold)
        return strategy_func(df, config=config)

    return strategy_func(df)


def run_backtest_cli(args: argparse.Namespace) -> None:
    try:
        df = fetch_price_data(args.symbol, args.start, args.end)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to load data: {exc}")
        return

    try:
        df = _run_strategy(
            args.strategy,
            df,
            fast=args.fast,
            slow=args.slow,
            rsi_period=args.rsi_period,
            rsi_threshold=args.rsi_threshold,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to compute strategy: {exc}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / timestamp
    try:
        result = run_backtest(
            df,
            starting_cash=args.cash,
            position_fraction=args.position_fraction,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            output_dir=output_dir,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Failed to run backtest: {exc}")
        return

    _plot_results(df, result, output_dir, args.symbol, show=args.plot)

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
    backtest_parser.add_argument("--strategy", default="sma_rsi", choices=STRATEGIES.keys(), help="Strategy name")
    backtest_parser.add_argument("--fast", type=int, default=20, help="Fast SMA window")
    backtest_parser.add_argument("--slow", type=int, default=50, help="Slow SMA window")
    backtest_parser.add_argument("--rsi-period", dest="rsi_period", type=int, default=14, help="RSI lookback length")
    backtest_parser.add_argument("--rsi-threshold", dest="rsi_threshold", type=float, default=50.0, help="RSI threshold required for a long signal")
    backtest_parser.add_argument("--cash", type=float, default=100_000, help="Starting portfolio value")
    backtest_parser.add_argument("--position-fraction", dest="position_fraction", type=float, default=1.0, help="Fraction of capital to deploy on signals (0-1)")
    backtest_parser.add_argument("--fee-bps", dest="fee_bps", type=float, default=1.0, help="Trading fee in basis points per trade")
    backtest_parser.add_argument("--slippage-bps", dest="slippage_bps", type=float, default=1.0, help="Slippage in basis points per trade")
    backtest_parser.add_argument("--plot", action="store_true", help="Show plot after saving")
    backtest_parser.set_defaults(func=run_backtest_cli)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
