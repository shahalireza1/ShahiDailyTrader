from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from trader.analytics.metrics import compute_metrics


def write_reports(
    symbol_frames: Dict[str, pd.DataFrame],
    equity_curve: pd.Series,
    trades: pd.DataFrame,
    starting_cash: float,
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    equity_path = output_dir / "equity_curve.csv"
    equity_curve.to_csv(equity_path, header=["equity"])
    paths["equity_curve"] = equity_path

    trades_path = output_dir / "trades.csv"
    trades.to_csv(trades_path, index=False)
    paths["trades"] = trades_path

    metrics = compute_metrics(equity_curve, trades, starting_cash)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    paths["metrics"] = metrics_path

    for symbol, frame in symbol_frames.items():
        frame.to_csv(output_dir / f"{symbol}_signals_and_prices.csv")

    return paths
