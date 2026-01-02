import pandas as pd

from trader.analytics.metrics import compute_metrics


def test_metrics_compute_basic_fields():
    equity = pd.Series([100_000, 102_000, 104_000], index=pd.date_range("2023-01-01", periods=3, freq="D"))
    trades = pd.DataFrame(
        [
            {"pnl": 0.02},
            {"pnl": -0.01},
            {"pnl": 0.03},
        ]
    )
    metrics = compute_metrics(equity, trades, starting_cash=100_000)
    assert metrics["total_return"] > 0
    assert metrics["num_trades"] == 3
    assert 0 <= metrics["win_rate"] <= 1
    assert metrics["expectancy"] != 0


def test_no_trades_means_no_performance():
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    equity = pd.Series([100_000, 101_000, 102_000], index=dates)

    metrics = compute_metrics(equity, trades=pd.DataFrame(), starting_cash=100_000)

    assert metrics["num_trades"] == 0
    assert metrics["total_return"] == 0
    assert metrics["sharpe"] == 0
    assert metrics["max_drawdown"] == 0
