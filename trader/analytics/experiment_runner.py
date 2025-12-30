from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd

from trader.core.engine import BacktestEngine
from trader.utils.config import load_config


DEFAULT_CONFIGS: List[Path] = [
    Path("config.yaml"),
    Path("configs/ensemble_momentum_biased.yaml"),
    Path("configs/ensemble_risk_budgeted.yaml"),
    Path("configs/ensemble_voting_momentum_overlay.yaml"),
]


def run_experiments(config_paths: Iterable[Path] | None = None) -> pd.DataFrame:
    configs = list(config_paths or DEFAULT_CONFIGS)
    records = []
    for path in configs:
        cfg = load_config(path)
        engine = BacktestEngine(cfg, enable_plots=False)
        result = engine.run()
        records.append(
            {
                "config": path.name,
                "cagr": result.metrics.get("cagr", 0.0),
                "max_drawdown": result.metrics.get("max_drawdown", 0.0),
                "sharpe": result.metrics.get("sharpe", 0.0),
                "exposure": result.metrics.get("exposure", 0.0),
            }
        )
    table = pd.DataFrame(records)
    print("\nExperiment Summary (CAGR, Drawdown, Sharpe, Exposure):")
    print(
        table.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )
    return table


if __name__ == "__main__":
    run_experiments()
