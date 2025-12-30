from pathlib import Path

import pandas as pd

from trader.analytics import experiment_runner
from trader.core.engine import EngineResult
from trader.utils.config import Config


def test_run_experiments_writes_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    outputs_root = tmp_path / "outputs"

    def fake_load_config(path):
        cfg = Config()
        cfg.strategy.name = path.stem
        cfg.output_dir = outputs_root
        return cfg

    class DummyEngine:
        def __init__(self, config, enable_plots):
            self.config = config
            self.enable_plots = enable_plots

        def run(self):
            run_dir = outputs_root / self.config.strategy.name
            plots_dir = run_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            for plot in ["equity_vs_benchmark", "drawdown", "monthly_heatmap"]:
                (plots_dir / f"{plot}.png").write_bytes(b"png")

            equity_curve = pd.Series([1.0, 1.1], index=pd.date_range("2024-01-01", periods=2, freq="D"))
            trades = pd.DataFrame([{"pnl": 0.01}])
            monthly_returns = pd.DataFrame([[0.01]], index=[2024], columns=["Jan"])
            metrics = {
                "total_return": 0.1,
                "cagr": 0.1 if self.config.strategy.name == "safe" else 0.02,
                "max_drawdown": -0.1 if self.config.strategy.name == "safe" else -0.3,
                "sharpe": 1.0,
                "sortino": 0.8,
                "win_rate": 0.6,
                "num_trades": 2,
                "exposure": 0.5 if self.config.strategy.name == "safe" else 0.9,
            }

            return EngineResult(
                equity_curve=equity_curve,
                trades=trades,
                metrics=metrics,
                symbol_frames={},
                benchmark_curve=equity_curve,
                monthly_returns=monthly_returns,
                output_dir=run_dir,
                plots_dir=plots_dir,
            )

    def fake_report(output_dir, *args, **kwargs):
        report_path = Path(output_dir) / "report.html"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        report_path.write_text("report")
        return report_path

    monkeypatch.setattr(experiment_runner, "load_config", fake_load_config)
    monkeypatch.setattr(experiment_runner, "BacktestEngine", DummyEngine)
    monkeypatch.setattr(experiment_runner, "build_html_report", fake_report)

    configs = [Path("safe.yaml"), Path("risky.yaml")]
    result = experiment_runner.run_experiments(configs)

    summary_path = Path("outputs/experiments/summary.csv")
    assert summary_path.exists()

    filtered_configs = set(result["config"])
    assert "safe.yaml" in filtered_configs
    assert "risky.yaml" not in filtered_configs
    assert result.loc[result["config"] == "safe.yaml", "highlight"].iloc[0]
