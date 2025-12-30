import pandas as pd

from trader.strategies.standard import SMACrossStrategy, SMARSIStrategy


def _mock_data() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    close = pd.Series(range(10), index=dates) + 1
    data = pd.DataFrame({
        "Open": close,
        "High": close + 1,
        "Low": close - 1,
        "Close": close,
        "Volume": 1,
    })
    return data


def test_sma_cross_signals_created():
    strategy = SMACrossStrategy(fast=2, slow=3)
    df = strategy.generate_signals(_mock_data())
    assert "signal" in df.columns
    assert df["signal"].isna().sum() == 0


def test_sma_rsi_respects_threshold():
    strategy = SMARSIStrategy(fast=2, slow=3, rsi_period=2, rsi_threshold=0)
    df = strategy.generate_signals(_mock_data())
    assert (df["signal"] >= 0).all()
    assert df["signal"].iloc[-1] == 1
