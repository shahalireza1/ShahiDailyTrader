from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from trader.analytics.reports import build_html_report
from trader.core.engine import BacktestEngine, EngineResult
from trader.data.loaders import DataLoader, DataRequest
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


def _run_engine(
    cfg: Config,
    enable_plots: bool,
    generate_html: bool,
    preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> EngineResult:
    engine = BacktestEngine(cfg, enable_plots=enable_plots, generate_html=generate_html, preloaded_data=preloaded_data)
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = output_root / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("ignore", module="pandas")

    records: List[Dict[str, Any]] = []
    combinations = list(_param_grid())
    total_runs = len(combinations)
    best_cagr = float("-inf")
    best_drawdown = float("-inf")
    start_time = time.perf_counter()

    csv_path = output_root / "full_results.csv"
    progress_path = output_root / "progress.json"
    fieldnames = [
        "position_fraction",
        "min_active_weight",
        "rebalance_band",
        "signal_persistence_days",
        "min_hold_days",
        "target_gross_exposure",
        "cagr",
        "max_drawdown",
        "sharpe",
        "total_return",
        "trades",
        "output_dir",
    ]

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_file.flush()

    progress_payload = {
        "runs_completed": 0,
        "total_runs": total_runs,
        "best_cagr": None,
        "best_max_drawdown": None,
        "last_run_timestamp": datetime.now().isoformat(),
    }
    progress_path.write_text(json.dumps(progress_payload, indent=2))

    loader = DataLoader()
    preloaded_data = loader.fetch_many(
        (DataRequest(symbol=s, start=base_config.start, end=base_config.end) for s in base_config.symbols)
    )

    with tqdm(total=total_runs, desc="Optimizing", unit="run") as pbar:
        with csv_path.open("a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            for idx, params in enumerate(combinations, start=1):
                cfg = _update_config(base_config, params)
                cfg.output_dir = output_root / f"run_{idx:04d}"
                result = _run_engine(cfg, enable_plots=False, generate_html=False, preloaded_data=preloaded_data)
                run_dir = result.output_dir
                record = _record_metrics(params, result, run_dir)
                records.append(record)
                writer.writerow(record)
                if idx % 10 == 0:
                    csv_file.flush()
                cagr = record["cagr"]
                max_drawdown = record["max_drawdown"]
                if cagr > best_cagr or (cagr == best_cagr and max_drawdown > best_drawdown):
                    best_cagr = cagr
                    best_drawdown = max_drawdown
                progress_payload = {
                    "runs_completed": idx,
                    "total_runs": total_runs,
                    "best_cagr": best_cagr,
                    "best_max_drawdown": best_drawdown,
                    "last_run_timestamp": datetime.now().isoformat(),
                }
                progress_path.write_text(json.dumps(progress_payload, indent=2))
                if idx % 50 == 0 or idx == total_runs:
                    elapsed = time.perf_counter() - start_time
                    print(
                        "completed {}/{},".format(idx, total_runs),
                        f"best CAGR={best_cagr:.4f},",
                        f"best DD={best_drawdown:.4f},",
                        f"elapsed={elapsed:.1f}s",
                    )
                pbar.set_description(f"Optimizing {idx}/{total_runs}")
                pbar.set_postfix(best_cagr=best_cagr, best_dd=best_drawdown)
                pbar.update(1)

    results_df = pd.DataFrame(records)
    results_df.sort_values(by=["cagr", "max_drawdown"], ascending=[False, False], inplace=True)

    results_path = output_root / "optimization_results.csv"
    results_df.to_csv(results_path, index=False)

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

    return results_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search optimizer for walk-forward backtests")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Base YAML config path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to store optimization artifacts",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top configs to regenerate reports for")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    csv_path = optimize(args.config, args.output_dir, args.top_k)
    print(f"Optimization complete. Results saved to {csv_path}")
