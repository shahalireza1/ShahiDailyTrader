from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from trader.core.engine import BacktestEngine
from trader.utils.config import Config, StrategyConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a deterministic multi-ticker backtest")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of ticker symbols")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--strategy",
        default="sma_rsi",
        choices=["sma_rsi", "sma_cross", "mean_reversion", "momentum", "breakout", "buy_hold", "ensemble"],
        help="Strategy to run",
    )
    parser.add_argument("--starting-cash", type=float, default=100_000.0, help="Starting portfolio value")
    parser.add_argument("--fees-bps", type=float, default=1.0, help="Commission in basis points")
    parser.add_argument("--slippage-bps", type=float, default=2.0, help="Slippage in basis points")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Base directory for reports")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.symbols = args.tickers
    cfg.start = args.start
    cfg.end = args.end
    cfg.mode = "backtest"
    cfg.starting_cash = args.starting_cash
    cfg.fees_bps = args.fees_bps
    cfg.slippage_bps = args.slippage_bps
    cfg.output_dir = args.output_dir
    cfg.strategy = StrategyConfig(name=args.strategy, parameters={})

    if args.strategy == "ensemble":
        cfg.strategies = [
            StrategyConfig(name="sma_rsi"),
            StrategyConfig(name="momentum", parameters={"lookback": 50, "threshold": 0}),
            StrategyConfig(name="breakout", parameters={"lookback": 20}),
        ]
        cfg.ensemble = {"mode": "equal_weight"}
    else:
        cfg.strategies = [cfg.strategy]

    return cfg


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    config = build_config(args)
    engine = BacktestEngine(config, enable_plots=False, generate_html=True)
    result = engine.run()

    print("Backtest complete!")
    print(f"Outputs directory: {result.output_dir.resolve()}")
    if result.report_path:
        print(f"HTML report: {result.report_path.resolve()}")


if __name__ == "__main__":
    main()
