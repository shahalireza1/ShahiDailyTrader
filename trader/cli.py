from __future__ import annotations

import argparse
from pathlib import Path

from trader.analytics.reports import build_html_report
from trader.core.engine import BacktestEngine, EngineResult
from trader.strategies import registry  # ensures strategies register
from trader.utils.config import Config, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quant research platform CLI")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to YAML config")
    parser.add_argument("--list-strategies", action="store_true", help="List available strategies")
    parser.add_argument("--run", choices=["backtest", "walkforward", "paper", "report"], help="Execution mode")
    parser.add_argument("--plot", action="store_true", help="Generate plots for the run")

    subparsers = parser.add_subparsers(dest="command")
    report_parser = subparsers.add_parser("report", help="Run a backtest and generate an HTML report")
    report_parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to YAML config")
    return parser


def _execute_backtest(
    config_path: Path, mode: str | None, enable_plots: bool, generate_report: bool
) -> EngineResult:
    config: Config = load_config(config_path)
    if mode:
        config.mode = mode

    if generate_report:
        config.mode = "backtest"

    engine = BacktestEngine(config, enable_plots=enable_plots or generate_report)
    result = engine.run()
    return result


def _print_metrics(result: EngineResult) -> None:
    print("\nRun complete!")
    print(f"Outputs saved to: {result.output_dir.resolve()}")
    if result.plots_dir:
        print(f"Plots: {result.plots_dir.resolve()}")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def run_cli(args: argparse.Namespace) -> None:
    command = args.command or "run"

    if command == "report":
        result = _execute_backtest(args.config, mode="backtest", enable_plots=True, generate_report=True)

        plots_dir = result.plots_dir or result.output_dir
        report_path = build_html_report(
            result.output_dir,
            result.metrics,
            result.monthly_returns,
            plots_dir / "equity_vs_benchmark.png",
            plots_dir / "drawdown.png",
            plots_dir / "monthly_heatmap.png",
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

    run_mode = args.run
    generate_report = run_mode == "report"
    result = _execute_backtest(args.config, mode=run_mode, enable_plots=args.plot, generate_report=generate_report)
    _print_metrics(result)

    if generate_report:
        plots_dir = result.plots_dir or result.output_dir
        report_path = build_html_report(
            result.output_dir,
            result.metrics,
            result.monthly_returns,
            plots_dir / "equity_vs_benchmark.png",
            plots_dir / "drawdown.png",
            plots_dir / "monthly_heatmap.png",
        )
        print(f"Report: {report_path.resolve()}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_cli(args)


if __name__ == "__main__":
    main()
