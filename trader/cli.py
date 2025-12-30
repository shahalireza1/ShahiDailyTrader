from __future__ import annotations

import argparse
from pathlib import Path

from trader.core.engine import BacktestEngine
from trader.strategies import registry  # ensures strategies register
from trader.utils.config import Config, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quant research platform CLI")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to YAML config")
    parser.add_argument("--list-strategies", action="store_true", help="List available strategies")
    parser.add_argument("--run", choices=["backtest", "walkforward", "paper"], help="Execution mode")
    return parser


def run_cli(args: argparse.Namespace) -> None:
    if args.list_strategies:
        print("Available strategies:")
        for name in registry.list_strategies():
            print(f" - {name}")
        return

    config: Config = load_config(args.config)
    if args.run:
        config.mode = args.run

    engine = BacktestEngine(config)
    result = engine.run()

    print("\nRun complete!")
    print(f"Outputs saved to: {result.output_dir.resolve()}")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_cli(args)


if __name__ == "__main__":
    main()
