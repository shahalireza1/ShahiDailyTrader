from __future__ import annotations

import argparse
import itertools
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from trader.analytics.reports import build_html_report
from trader.core.engine import BacktestEngine, EngineResult
from trader.utils.config import Config, load_config


def _param_grid() -> Iterable[Tuple[float, float, float, int, int, Optional[float]]]:
    position_fractions = [0.005, 0.015]
    min_active_weights = [0.00, 0.05, 0.10]
    rebalance_bands = [0.02, 0.05, 0.10]
    signal_persistence_days = [1, 3, 5]
    holding_days = [1, 5, 10]
    target_gross_exposures: List[Optional[float]] = [None, 0.2, 0.5, 0.8, 1.0]

    return itertools.product(
        position_fractions,
        min_active_weights,
        rebalance_bands,
        signal_persistence_days,
        holding_days,
        target_gross_exposures,
    )


def _update_config(base: Config, params: Tuple[float, float, float, int, int, Optional[float]]) -> Config:
    position_fraction, min_active_weight, rebalance_band, persistence_days, hold_days, target_gross = params
    cfg = deepcopy(base)
    cfg.mode = "walkforward"
    cfg.risk.position_fraction = position_fraction
    cfg.risk.min_active_weight = min_active_weight
    cfg.risk.rebalance_band = rebalance_band
    cfg.risk.signal_persistence_days = persistence_days
    cfg.risk.min_hold_days = hold_days
    cfg.risk.target_gross_exposure = target_gross
    if target_gross is not None:
        cfg.risk.max_gross_exposure = max(cfg.risk.max_gross_exposure, target_gross)
    return cfg


def _run_engine(cfg: Config, enable_plots: bool, generate_html: bool) -> EngineResult:
    engine = BacktestEngine(cfg, enable_plots=enable_plots, generate_html=generate_html)
    return engine.run()


def _record_metrics(
    params: Tuple[float, float, float, int, int, Optional[float]], result: EngineResult, run_dir: Path
) -> Dict[str, Any]:
    position_fraction, min_active_weight, rebalance_band, persistence_days, hold_days, target_gross = params
    metrics = result.metrics
    return {
        "position_fraction": position_fraction,
        "min_active_weight": min_active_weight,
        "rebalance_band": rebalance_band,
        "signal_persistence_days": persistence_days,
        "min_hold_days": hold_days,
        "target_gross_exposure": target_gross,
        "cagr": metrics.get("cagr", 0.0),
        "max_drawdown": metrics.get("max_drawdown", 0.0),
        "sharpe": metrics.get("sharpe", 0.0),
        "total_return": metrics.get("total_return", 0.0),
        "trades": metrics.get("num_trades", 0),
        "output_dir": str(run_dir),
    }


def _generate_report(result: EngineResult) -> Path:
    plots_dir = result.plots_dir or result.output_dir
    return build_html_report(
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
        diagnostics=(result.summary or {}).get("diagnostics") if result.summary else None,
    )


def optimize(config_path: Path, output_root: Path, top_k: int = 5) -> Path:
    base_config = load_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    combinations = list(_param_grid())

    for idx, params in enumerate(combinations, start=1):
        cfg = _update_config(base_config, params)
        cfg.output_dir = output_root / f"run_{idx:04d}"
        result = _run_engine(cfg, enable_plots=False, generate_html=False)
        run_dir = result.output_dir
        records.append(_record_metrics(params, result, run_dir))

    results_df = pd.DataFrame(records)
    results_df.sort_values(by=["cagr", "max_drawdown"], ascending=[False, False], inplace=True)

    csv_path = output_root / "optimization_results.csv"
    results_df.to_csv(csv_path, index=False)

    top_rows = results_df.head(top_k)
    for rank, row in enumerate(top_rows.itertuples(index=False), start=1):
        params = (
            row.position_fraction,
            row.min_active_weight,
            row.rebalance_band,
            int(row.signal_persistence_days),
            int(row.min_hold_days),
            row.target_gross_exposure if pd.notna(row.target_gross_exposure) else None,
        )
        cfg = _update_config(base_config, params)
        cfg.output_dir = output_root / f"top_{rank:02d}"
        detailed_result = _run_engine(cfg, enable_plots=True, generate_html=True)
        if detailed_result.report_path is None:
            detailed_result.report_path = _generate_report(detailed_result)

    return csv_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search optimizer for walk-forward backtests")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Base YAML config path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/optimization"),
        help="Directory to store optimization artifacts",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top configs to regenerate reports for")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    csv_path = optimize(args.config, args.output_dir, args.top_k)
    print(f"Optimization complete. Results saved to {csv_path}")
