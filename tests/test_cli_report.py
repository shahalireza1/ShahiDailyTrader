import argparse
from pathlib import Path

import pandas as pd

from trader import cli
from trader.core.engine import EngineResult
from trader.utils.config import Config


def test_report_cli_creates_html(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    config_path = tmp_path / "config.yaml"
    config_path.write_text("symbols: []\n")

    def fake_load_config(path):
        cfg = Config()
        cfg.output_dir = output_dir
        return cfg

    class DummyEngine:
        def __init__(self, config, enable_plots):
            self.config = config
            self.enable_plots = enable_plots

        def run(self):
            run_dir = output_dir / "dummy_run"
            plots_dir = run_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            for name in ["equity_vs_benchmark", "drawdown", "monthly_heatmap"]:
                (plots_dir / f"{name}.png").write_bytes(b"png")

            metrics = {"total_return": 0.05}
            equity_curve = pd.Series([1.0, 1.1], index=pd.date_range("2024-01-01", periods=2, freq="D"))
            trades = pd.DataFrame([{"pnl": 0.01}])
            monthly_returns = pd.DataFrame([[0.01]], index=[2024], columns=["Jan"])
            benchmark_curve = equity_curve.copy()

            return EngineResult(
                equity_curve=equity_curve,
                trades=trades,
                metrics=metrics,
                symbol_frames={},
                benchmark_curve=benchmark_curve,
                monthly_returns=monthly_returns,
                output_dir=run_dir,
                plots_dir=plots_dir,
            )

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "BacktestEngine", DummyEngine)

    args = argparse.Namespace(config=config_path, list_strategies=False, run="report", plot=False, command=None)
    cli.run_cli(args)

    report_path = output_dir / "dummy_run" / "report.html"
    assert report_path.exists()


def test_report_cli_supports_ensemble_config(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"

    real_load_config = cli.load_config

    def wrapped_load_config(path):
        cfg = real_load_config(path)
        cfg.output_dir = output_dir
        return cfg

    class DummyEngine:
        def __init__(self, config, enable_plots):
            self.config = config
            self.enable_plots = enable_plots

        def run(self):
            run_dir = output_dir / "dummy_run"
            plots_dir = run_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            for name in [
                "equity_vs_benchmark",
                "drawdown",
                "monthly_heatmap",
                "exposure",
                "rolling_sharpe",
                "strategy_contribution",
                "spy_comparison",
            ]:
                (plots_dir / f"{name}.png").write_bytes(b"png")

            metrics = {"total_return": 0.05}
            equity_curve = pd.Series([1.0, 1.1], index=pd.date_range("2024-01-01", periods=2, freq="D"))
            trades = pd.DataFrame([{"pnl": 0.01}])
            monthly_returns = pd.DataFrame([[0.01]], index=[2024], columns=["Jan"])
            benchmark_curve = equity_curve.copy()

            return EngineResult(
                equity_curve=equity_curve,
                trades=trades,
                metrics=metrics,
                symbol_frames={},
                benchmark_curve=benchmark_curve,
                monthly_returns=monthly_returns,
                output_dir=run_dir,
                plots_dir=plots_dir,
            )

    monkeypatch.setattr(cli, "load_config", wrapped_load_config)
    monkeypatch.setattr(cli, "BacktestEngine", DummyEngine)

    args = argparse.Namespace(
        config=Path("configs/ensemble_momentum_biased.yaml"),
        list_strategies=False,
        run="report",
        plot=False,
        command=None,
    )
    cli.run_cli(args)

    report_path = output_dir / "dummy_run" / "report.html"
    assert report_path.exists()
