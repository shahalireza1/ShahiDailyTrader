from __future__ import annotations

import base64
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
    metrics: Dict[str, float] | None = None,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    equity_path = output_dir / "equity_curve.csv"
    equity_curve.to_csv(equity_path, header=["equity"])
    paths["equity_curve"] = equity_path

    trades_path = output_dir / "trades.csv"
    trades.to_csv(trades_path, index=False)
    paths["trades"] = trades_path

    computed_metrics = metrics or compute_metrics(equity_curve, trades, starting_cash)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(computed_metrics, indent=2))
    paths["metrics"] = metrics_path

    for symbol, frame in symbol_frames.items():
        frame.to_csv(output_dir / f"{symbol}_signals_and_prices.csv")

    return paths


def _encode_img(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def build_html_report(
    output_dir: Path,
    metrics: Dict[str, float],
    monthly_returns: pd.DataFrame,
    equity_vs_benchmark: Path,
    drawdown_plot: Path,
    heatmap_plot: Path,
    strategy_attribution: pd.DataFrame | None = None,
) -> Path:
    metrics_df = pd.DataFrame(metrics, index=["value"]).T
    metrics_html = metrics_df.to_html(float_format=lambda x: f"{x:.4f}")

    monthly_pct = monthly_returns.copy() * 100
    monthly_html = monthly_pct.to_html(float_format=lambda x: f"{x:.2f}%")

    attribution_html = ""
    if strategy_attribution is not None and not strategy_attribution.empty:
        attribution_html = "<h2>Strategy Attribution</h2>" + strategy_attribution.to_html(
            float_format=lambda x: f"{x:.4f}"
        )

    html = f"""
    <html>
      <head>
        <meta charset='UTF-8'>
        <title>Trading Report</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 20px; }}
          h1, h2 {{ color: #1f2937; }}
          table {{ border-collapse: collapse; margin-bottom: 20px; }}
          table, th, td {{ border: 1px solid #e5e7eb; padding: 6px 10px; }}
          th {{ background: #f3f4f6; }}
          img {{ max-width: 100%; height: auto; }}
        </style>
      </head>
      <body>
        <h1>Trading Report</h1>
        <h2>Key Metrics</h2>
        {metrics_html}
        <h2>Equity vs Buy & Hold</h2>
        <img src="data:image/png;base64,{_encode_img(equity_vs_benchmark)}" alt="Equity vs Benchmark" />
        <h2>Drawdown</h2>
        <img src="data:image/png;base64,{_encode_img(drawdown_plot)}" alt="Drawdown" />
        <h2>Monthly Returns</h2>
        <img src="data:image/png;base64,{_encode_img(heatmap_plot)}" alt="Monthly Returns Heatmap" />
        {monthly_html}
        {attribution_html}
      </body>
    </html>
    """
    report_path = output_dir / "report.html"
    report_path.write_text(html)
    return report_path
