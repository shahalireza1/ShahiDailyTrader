import pandas as pd

from trader.analytics.metrics import compute_metrics
from trader.core.engine import BacktestEngine
from trader.core.portfolio import Portfolio
from trader.core.risk import PositionSizingConfig
from trader.utils.config import Config


def _make_portfolio():
    risk_cfg = PositionSizingConfig(
        mode="fixed_fraction",
        fraction=1.0,
        kelly_safety=1.0,
        target_volatility=None,
        vol_lookback=5,
    )
    return Portfolio(
        starting_cash=100_000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        risk_config=risk_cfg,
        max_drawdown=1.0,
        max_drawdown_stop=None,
        drawdown_safe_fraction=0.0,
        max_gross_exposure=1.0,
        max_position_per_symbol=1.0,
        trade_cooldown_days=0,
    )


def test_trades_exist_when_positions_change():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame({"Close": [100, 102, 103, 104], "signal": [0.0, 1.0, 1.0, 0.0]}, index=dates)
    portfolio = _make_portfolio()
    per_symbol_results = {"AAA": portfolio.backtest_symbol(df, 1.0)}

    result = portfolio.combine(per_symbol_results)

    assert (result.positions.abs().sum(axis=1) > 0).any()
    assert not result.trades.empty


def test_flat_equity_when_no_trades_and_no_positions():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame({"Close": [100, 100, 100, 100], "signal": [0.0, 0.0, 0.0, 0.0]}, index=dates)
    portfolio = _make_portfolio()
    per_symbol_results = {"AAA": portfolio.backtest_symbol(df, 1.0)}

    result = portfolio.combine(per_symbol_results)
    metrics = compute_metrics(result.equity_curve, result.trades, portfolio.starting_cash)

    engine = BacktestEngine(Config(), enable_plots=False, generate_html=False)
    summary = engine._build_run_summary(
        metrics,
        result.equity_curve,
        result.trades,
        result.positions,
        pd.Series(dtype=float),
        per_symbol_results,
    )

    assert result.trades.empty
    assert result.equity_curve.diff().abs().max() == 0
    assert summary["return_source"] in {"positions_only", "benchmark_only"}
    if summary["return_source"] == "positions_only":
        assert summary["equity_change_days"] == 0
