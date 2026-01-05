import pandas as pd

from trader.analytics.metrics import compute_metrics


def test_zero_trades_zeroes_out_metrics():
    # Equity curve that would imply positive performance, but no trades were executed.
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    equity_curve = pd.Series([100_000, 101_000, 102_000, 103_000, 104_000], index=dates)

    metrics = compute_metrics(equity_curve, trades=pd.DataFrame(), starting_cash=100_000)

    assert metrics["num_trades"] == 0
    assert metrics["total_return"] == 0.0
    assert metrics["cagr"] == 0.0
    assert metrics["sharpe"] == 0.0
    assert metrics["sortino"] == 0.0
    assert metrics["max_drawdown"] == 0.0
