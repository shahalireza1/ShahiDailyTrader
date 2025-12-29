from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

DATA_CACHE_DIR = Path("data_cache")


def _format_cache_name(symbol: str, start: str, end: str) -> Path:
    safe_symbol = symbol.replace("/", "-")
    safe_start = start.replace("-", "")
    safe_end = end.replace("-", "")
    return DATA_CACHE_DIR / f"{safe_symbol}_{safe_start}_{safe_end}.csv"


def fetch_price_data(symbol: str, start: str, end: str, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load historical data with on-disk caching.

    Data is pulled from yfinance and cached as a CSV under ``data_cache/`` to
    avoid repeated downloads. Adjusted close is used when available to avoid
    dividend/ split distortions. The returned DataFrame always contains the
    columns ``Open``, ``High``, ``Low``, ``Close``, and ``Volume`` with a
    DatetimeIndex.
    """

    cache_root = cache_dir or DATA_CACHE_DIR
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_file = cache_root / _format_cache_name(symbol, start, end).name

    if cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    else:
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            raise RuntimeError(f"No data downloaded for {symbol} between {start} and {end}.")
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        df.to_csv(cache_file)

    # Prefer adjusted close when available
    if "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Downloaded data missing columns: {', '.join(missing)}")

    df = df[required_cols]
    df.dropna(subset=["Close"], inplace=True)
    return df


def get_latest_data(symbol: str, window: int = 200) -> pd.DataFrame:
    """Convenience helper for quick smoke tests."""

    df = fetch_price_data(symbol, start="2023-01-01", end="2024-12-31")
    return df.tail(window)
