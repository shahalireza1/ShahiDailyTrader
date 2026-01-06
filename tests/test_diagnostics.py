import pandas as pd

from trader.analytics.metrics import compute_metrics
from trader.core.diagnostics import PipelineDiagnostics
from trader.core.engine import BacktestEngine
from trader.utils.config import Config


def _build_summary(engine: BacktestEngine, positions: pd.DataFrame, symbol_frame: pd.DataFrame, costs: pd.Series):
    equity = pd.Series(100000 + pd.Series(range(len(positions))), index=positions.index)
    trades = pd.DataFrame([{"pnl": 1.0}])
    metrics = compute_metrics(equity, trades, engine.config.starting_cash, transaction_costs=costs)
    return engine._build_run_summary(
        metrics,
        equity,
        trades,
        positions,
        pd.Series(dtype=float),
        {"AAA": symbol_frame},
        gross_exposure=positions.abs().sum(axis=1),
        transaction_costs=costs,
        diagnostics=PipelineDiagnostics(),
    )


def test_summary_contains_new_diagnostics(tmp_path):
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    positions = pd.DataFrame({"AAA": [0.0, 0.2, 0.2, 0.0, 0.0]}, index=dates)
    symbol_frame = pd.DataFrame(
        {
            "signal": [0.0, 1.0, 1.0, 0.0, 0.0],
            "final_position": positions["AAA"],
            "target_weight_pre_cap": positions["AAA"],
            "Close": 100,
            "return": 0.0,
        },
        index=dates,
    )
    costs = pd.Series([1.0] * len(dates), index=dates)
    engine = BacktestEngine(Config(symbols=["AAA"], output_dir=tmp_path), enable_plots=False, generate_html=False)

    summary = _build_summary(engine, positions, symbol_frame, costs)
    diagnostics = summary.get("diagnostics", {})

    expected_keys = {
        "pct_days_in_cash",
        "pct_days_in_market",
        "avg_gross_exposure",
        "median_gross_exposure",
        "p90_gross_exposure",
        "signal_activity",
        "avg_target_weight_when_active",
        "block_reasons",
        "transaction_costs_total",
        "transaction_costs_bps_of_equity",
    }
    assert expected_keys.issubset(diagnostics.keys())
    assert "AAA" in diagnostics["signal_activity"].get("per_symbol", {})
    assert "AAA" in diagnostics["avg_target_weight_when_active"].get("per_symbol", {})


def test_high_exposure_config_increases_gross_exposure(tmp_path):
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    baseline_positions = pd.DataFrame({"AAA": [0.15] * len(dates)}, index=dates)
    high_positions = pd.DataFrame({"AAA": [0.6] * len(dates)}, index=dates)
    baseline_frame = pd.DataFrame(
        {"signal": 1.0, "final_position": baseline_positions["AAA"], "target_weight_pre_cap": baseline_positions["AAA"]},
        index=dates,
    )
    high_frame = pd.DataFrame(
        {"signal": 1.0, "final_position": high_positions["AAA"], "target_weight_pre_cap": high_positions["AAA"]},
        index=dates,
    )
    costs = pd.Series(0.5, index=dates)

    baseline_engine = BacktestEngine(Config(symbols=["AAA"], output_dir=tmp_path), enable_plots=False, generate_html=False)
    high_engine = BacktestEngine(Config(symbols=["AAA"], output_dir=tmp_path), enable_plots=False, generate_html=False)

    baseline_summary = _build_summary(baseline_engine, baseline_positions, baseline_frame, costs)
    high_summary = _build_summary(high_engine, high_positions, high_frame, costs)

    baseline_exposure = baseline_summary["diagnostics"]["avg_gross_exposure"]
    high_exposure = high_summary["diagnostics"]["avg_gross_exposure"]

    assert high_exposure > baseline_exposure
