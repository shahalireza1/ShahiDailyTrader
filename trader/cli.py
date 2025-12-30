from __future__ import annotations

import argparse
from pathlib import Path

from trader.analytics.reports import build_html_report
from trader.core.engine import BacktestEngine
from trader.strategies import registry  # ensures strategies register
from trader.utils.config import Config, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quant research platform CLI")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to YAML config")
    parser.add_argument("--list-strategies", action="store_true", help="List available strategies")
    parser.add_argument("--run", choices=["backtest", "walkforward", "paper"], help="Execution mode")

    subparsers = parser.add_subparsers(dest="command")
    report_parser = subparsers.add_parser("report", help="Run a backtest and generate an HTML report")
    report_parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to YAML config")
    return parser


def run_cli(args: argparse.Namespace) -> None:
    command = args.command or "run"
    config_path: Path = args.config

    if command == "report":
        config = load_config(args.config)
        config.mode = "backtest"
        engine = BacktestEngine(config)
        result = engine.run()

        report_path = build_html_report(
            result.output_dir,
            result.metrics,
            result.monthly_returns,
            result.output_dir / "equity_vs_benchmark.png",
            result.output_dir / "drawdown.png",
            result.output_dir / "monthly_heatmap.png",
        )
        print("\nReport ready!")
        print(f"Outputs saved to: {result.output_dir.resolve()}")
        print(f"Report: {report_path.resolve()}")
        return

    if args.list_strategies:
        print("Available strategies:")
        for name in registry.list_strategies():
            print(f" - {name}")
        return

    config: Config = load_config(config_path)
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
