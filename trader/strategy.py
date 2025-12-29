from dataclasses import dataclass
from typing import Optional

import pandas as pd


def _validate_sma_windows(fast: int, slow: int) -> None:
    if fast <= 0 or slow <= 0:
        raise ValueError("SMA windows must be positive integers.")
    if fast >= slow:
        raise ValueError("The fast window should be smaller than the slow window to form a crossover.")


def sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """Simple SMA crossover strategy.

    Produces a ``signal`` column: 1 for long, 0 for flat. The signal is aligned
    with the close of the same day, so backtests must shift the signal by one to
    avoid lookahead bias.
    """

    _validate_sma_windows(fast, slow)

    df = df.copy()
    df["fast_sma"] = df["Close"].rolling(fast, min_periods=fast).mean()
    df["slow_sma"] = df["Close"].rolling(slow, min_periods=slow).mean()
    df["signal"] = (df["fast_sma"] > df["slow_sma"]).astype(int)
    df["signal"] = df["signal"].ffill().fillna(0)
    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    if period <= 0:
        raise ValueError("RSI period must be a positive integer.")

    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


@dataclass
class SMARsiConfig:
    fast: int = 20
    slow: int = 50
    rsi_period: int = 14
    rsi_threshold: float = 50.0


def sma_rsi_filter(df: pd.DataFrame, config: Optional[SMARsiConfig] = None) -> pd.DataFrame:
    """SMA crossover with an RSI momentum filter.

    A long signal is produced when the fast SMA is above the slow SMA **and**
    the RSI is greater than ``rsi_threshold``. The signal is aligned with the
    close of the same day, so callers must shift by one period to avoid
    lookahead bias during backtests.
    """

    cfg = config or SMARsiConfig()
    _validate_sma_windows(cfg.fast, cfg.slow)

    df = df.copy()
    df["fast_sma"] = df["Close"].rolling(cfg.fast, min_periods=cfg.fast).mean()
    df["slow_sma"] = df["Close"].rolling(cfg.slow, min_periods=cfg.slow).mean()
    df["rsi"] = _rsi(df["Close"], period=cfg.rsi_period)

    df["signal"] = ((df["fast_sma"] > df["slow_sma"]) & (df["rsi"] > cfg.rsi_threshold)).astype(int)
    df["signal"] = df["signal"].ffill().fillna(0)
    return df
