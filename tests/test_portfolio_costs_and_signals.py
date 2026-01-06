import pandas as pd
import pytest

from trader.analytics.metrics import compute_metrics
from trader.core.portfolio import Portfolio
from trader.core.risk import PositionSizingConfig


@pytest.fixture()
def risk_config():
    return PositionSizingConfig(
        mode="fixed_fraction",
        fraction=1.0,
        kelly_safety=1.0,
        target_volatility=None,
        vol_lookback=5,
    )


def _portfolio(fee_bps: float, slippage_bps: float, risk_cfg: PositionSizingConfig) -> Portfolio:
    return Portfolio(
        starting_cash=100_000.0,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        risk_config=risk_cfg,
        max_drawdown=1.0,
        max_drawdown_stop=None,
        drawdown_safe_fraction=0.0,
        max_gross_exposure=1.0,
        max_position_per_symbol=1.0,
        trade_cooldown_days=0,
        rebalance_band=0.0,
        signal_frequency="daily",
        signal_persistence_days=0,
    )


def test_transaction_costs_reduce_equity(risk_config: PositionSizingConfig):
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame({"Close": [100, 102, 101, 103], "signal": [0.0, 1.0, 0.0, 1.0]}, index=dates)

    costed = _portfolio(25.0, 25.0, risk_config)
    clean = _portfolio(0.0, 0.0, risk_config)

    costed_result = costed.combine({"AAA": costed.backtest_symbol(df, 1.0)})
    clean_result = clean.combine({"AAA": clean.backtest_symbol(df, 1.0)})

    assert costed_result.transaction_costs.sum() > 0
    assert costed_result.equity_curve.iloc[-1] < clean_result.equity_curve.iloc[-1]

    metrics = compute_metrics(
        costed_result.equity_curve,
        costed_result.trades,
        costed.starting_cash,
        transaction_costs=costed_result.transaction_costs,
    )
    assert metrics["fees_paid"] > 0
    assert metrics["transaction_costs"] > 0


def test_signals_shift_and_allow_shorts(risk_config: PositionSizingConfig):
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame({"Close": [100, 99, 98, 99], "signal": [1.0, -1.0, -1.0, 1.0]}, index=dates)

    portfolio = _portfolio(0.0, 0.0, risk_config)
    result = portfolio.backtest_symbol(df, 1.0)

    assert result.loc[dates[0], "position"] == 0  # shifted by one day
    assert result["position"].min() < 0  # short exposure supported
