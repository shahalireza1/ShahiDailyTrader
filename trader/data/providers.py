from __future__ import annotations

from typing import Dict, List

import pandas as pd


class SentimentProvider:
    """Placeholder sentiment/news/earnings signal provider."""

    def load_sentiment(self, symbols: List[str]) -> Dict[str, pd.Series]:  # pragma: no cover - placeholder
        return {symbol: pd.Series(dtype=float) for symbol in symbols}


class MLSignalProvider:
    """Placeholder for ML-driven alpha inputs."""

    def predict(self, data: pd.DataFrame) -> pd.Series:  # pragma: no cover - placeholder
        return pd.Series(0, index=data.index)
