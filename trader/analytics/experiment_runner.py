from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from trader.analytics.reports import build_html_report
from trader.core.engine import BacktestEngine
from trader.utils.config import load_config


DEFAULT_CONFIGS: List[Path] = [
    Path("config.yaml"),
    Path("configs/ensemble_momentum_biased.yaml"),
    Path("configs/ensemble_risk_budgeted.yaml"),
    Path("configs/ensemble_vote.yaml"),
    Path("configs/ensemble_voting_momentum_overlay.yaml"),
]


def run_experiments(
    config_paths: Iterable[Path] | None = None,
    max_drawdown_limit: float = -0.15,
    exposure_limit: float = 0.80,
    highlight_cagr: float = 0.08,
) -> pd.DataFrame:
    configs = list(config_paths or DEFAULT_CONFIGS)
    records = []
    for path in configs:
        cfg = load_config(path)
        engine = BacktestEngine(cfg, enable_plots=True)
        result = engine.run()
        plots_dir = result.plots_dir or result.output_dir
        build_html_report(
            result.output_dir,
            result.metrics,
            result.monthly_returns,
            plots_dir / "equity_vs_benchmark.png",
            plots_dir / "drawdown.png",
            plots_dir / "monthly_heatmap.png",
            result.strategy_attribution,
            exposure_plot=plots_dir / "exposure.png",
            rolling_sharpe_plot=plots_dir / "rolling_sharpe.png",
            strategy_contribution_plot=plots_dir / "strategy_contribution.png",
            spy_comparison_plot=plots_dir / "spy_comparison.png",
        )
        records.append(
            {
                "config": path.name,
                "cagr": result.metrics.get("cagr", 0.0),
                "total_return": result.metrics.get("total_return", 0.0),
                "max_drawdown": result.metrics.get("max_drawdown", 0.0),
                "sharpe": result.metrics.get("sharpe", 0.0),
                "sortino": result.metrics.get("sortino", 0.0),
                "win_rate": result.metrics.get("win_rate", 0.0),
                "num_trades": result.metrics.get("num_trades", 0),
                "exposure": result.metrics.get("exposure", 0.0),
            }
        )
    table = pd.DataFrame(records)
    table["within_risk"] = (table["max_drawdown"] >= max_drawdown_limit) & (table["exposure"] <= exposure_limit)
    table["highlight"] = table["within_risk"] & (table["cagr"] >= highlight_cagr)

    filtered = table[table["within_risk"]].sort_values("cagr", ascending=False)

    experiments_dir = Path("outputs/experiments")
    experiments_dir.mkdir(parents=True, exist_ok=True)
    summary_path = experiments_dir / "summary.csv"
    filtered.to_csv(summary_path, index=False)

    print("\nExperiment Summary (risk-filtered, sorted by CAGR):")
    print(
        filtered.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    print(f"\nSummary saved to: {summary_path.resolve()}")
    return filtered


if __name__ == "__main__":
    run_experiments()
