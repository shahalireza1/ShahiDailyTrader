import pandas as pd


def sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    """Simple SMA crossover strategy.

    Produces a ``signal`` column: 1 for long, 0 for flat. The signal is aligned
    with the close of the same day, so backtests must shift the signal by one to
    avoid lookahead bias.
    """

    if fast <= 0 or slow <= 0:
        raise ValueError("SMA windows must be positive integers.")
    if fast >= slow:
        raise ValueError("The fast window should be smaller than the slow window to form a crossover.")

    df = df.copy()
    df["fast_sma"] = df["Close"].rolling(fast, min_periods=fast).mean()
    df["slow_sma"] = df["Close"].rolling(slow, min_periods=slow).mean()
    df["signal"] = (df["fast_sma"] > df["slow_sma"]).astype(int)
    df["signal"] = df["signal"].ffill().fillna(0)
    return df
