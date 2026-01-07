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

    def _flatten_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.columns, pd.MultiIndex):
            return df

        price_fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        candidate_levels = []
        for level in range(df.columns.nlevels):
            level_values = set(df.columns.get_level_values(level))
            matches = price_fields & level_values
            if matches:
                candidate_levels.append((level, len(matches)))

        if candidate_levels:
            price_level = sorted(candidate_levels, key=lambda item: (-item[1], item[0]))[0][0]
            df.columns = df.columns.get_level_values(price_level)
        else:
            df.columns = df.columns.get_level_values(-1)

        return df

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._flatten_columns(df.copy())

        if "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Downloaded data missing columns: {', '.join(missing)}")

        cleaned = df[required_cols].dropna(subset=["Close"])
        return cleaned

    def fetch(self, request: DataRequest) -> pd.DataFrame:
        cache_file = self._cache_path(request)
        from_cache = cache_file.exists()

        if from_cache:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        else:
            df = yf.download(request.symbol, start=request.start, end=request.end, progress=False, auto_adjust=False)
            if df.empty:
                raise RuntimeError(
                    f"No data downloaded for {request.symbol} between {request.start} and {request.end}."
                )
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

        normalized = self._normalize(df)

        if not from_cache:
            normalized.to_csv(cache_file)

        return normalized

    def fetch_many(
        self,
        requests: Iterable[DataRequest],
        preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, pd.DataFrame]:
        results: Dict[str, pd.DataFrame] = {}
        for req in requests:
            if preloaded_data and req.symbol in preloaded_data:
                frame = preloaded_data[req.symbol]
                if not isinstance(frame.index, pd.DatetimeIndex):
                    frame = frame.copy()
                    frame.index = pd.to_datetime(frame.index)
                start = pd.to_datetime(req.start)
                end = pd.to_datetime(req.end)
                results[req.symbol] = frame.loc[start:end].copy()
            else:
                results[req.symbol] = self.fetch(req)
        return results
