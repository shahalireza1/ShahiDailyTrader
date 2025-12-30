from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd
import yfinance as yf


@dataclass
class DataRequest:
    symbol: str
    start: str
    end: str


class DataLoader:
    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or Path("data_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, request: DataRequest) -> Path:
        safe_symbol = request.symbol.replace("/", "-")
        safe_start = request.start.replace("-", "")
        safe_end = request.end.replace("-", "")
        return self.cache_dir / f"{safe_symbol}_{safe_start}_{safe_end}.csv"

    def fetch(self, request: DataRequest) -> pd.DataFrame:
        cache_file = self._cache_path(request)
        if cache_file.exists():
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            df = yf.download(request.symbol, start=request.start, end=request.end, progress=False, auto_adjust=False)
            if df.empty:
                raise RuntimeError(
                    f"No data downloaded for {request.symbol} between {request.start} and {request.end}."
                )
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            df.to_csv(cache_file)

        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Downloaded data missing columns: {', '.join(missing)}")

        cleaned = df[required_cols].dropna(subset=["Close"])
        return cleaned

    def fetch_many(self, requests: Iterable[DataRequest]) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        for req in requests:
            results[req.symbol] = self.fetch(req)
        return results
