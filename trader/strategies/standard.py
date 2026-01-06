from __future__ import annotations

from dataclasses import dataclass
from dataclasses import dataclass

import pandas as pd

from trader.signals.indicators import rsi, sma, zscore
from trader.strategies.base import Strategy, register_strategy


@dataclass
class SMARSIParameters:
    fast: int = 20
    slow: int = 50
    rsi_period: int = 14
    rsi_threshold: float = 50.0


@register_strategy
class SMARSIStrategy(Strategy):
    name = "sma_rsi"
    description = "SMA crossover with RSI confirmation"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        params = SMARSIParameters(**{**SMARSIParameters().__dict__, **self.parameters})
        df = data.copy()
        df["fast_sma"] = sma(df["Close"], params.fast)
        df["slow_sma"] = sma(df["Close"], params.slow)
        df["rsi"] = rsi(df["Close"], params.rsi_period)
        df["signal"] = ((df["fast_sma"] > df["slow_sma"]) & (df["rsi"] > params.rsi_threshold)).astype(int)
        df["signal"] = _apply_trend_filter(df, df["signal"]).ffill().fillna(0)
        return df


@register_strategy
class SMACrossStrategy(Strategy):
    name = "sma_cross"
    description = "Simple SMA crossover"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        fast = int(self.parameters.get("fast", 20))
        slow = int(self.parameters.get("slow", 50))
        if fast <= 0 or slow <= 0:
            raise ValueError("SMA windows must be positive")
        if fast >= slow:
            raise ValueError("Fast window must be smaller than slow window")
        df = data.copy()
        df["fast_sma"] = sma(df["Close"], fast)
        df["slow_sma"] = sma(df["Close"], slow)
        df["signal"] = (df["fast_sma"] > df["slow_sma"]).astype(int)
        df["signal"] = _apply_trend_filter(df, df["signal"]).ffill().fillna(0)
        return df


@register_strategy
class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    description = "Z-score based mean reversion"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        lookback = int(self.parameters.get("lookback", 20))
        entry_z = float(self.parameters.get("entry_z", -1.0))
        exit_z = float(self.parameters.get("exit_z", 0.0))
        df = data.copy()
        df["z"] = zscore(df["Close"], lookback)
        df["signal"] = 0
        df.loc[df["z"] <= entry_z, "signal"] = 1
        df.loc[df["z"] >= exit_z, "signal"] = 0
        df["signal"] = _apply_trend_filter(df, df["signal"]).ffill().fillna(0)
        return df


@register_strategy
class MomentumStrategy(Strategy):
    name = "momentum"
    description = "Momentum breakout using lookback returns"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        lookback = int(self.parameters.get("lookback", 50))
        threshold = float(self.parameters.get("threshold", 0.0))
        df = data.copy()
        df["momentum"] = df["Close"].pct_change(periods=lookback)
        df["signal"] = (df["momentum"] > threshold).astype(int)
        df["signal"] = _apply_trend_filter(df, df["signal"]).ffill().fillna(0)
        return df


@register_strategy
class BreakoutStrategy(Strategy):
    name = "breakout"
    description = "Donchian-style breakout strategy"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        lookback = int(self.parameters.get("lookback", 20))
        df = data.copy()
        df["rolling_high"] = df["High"].rolling(lookback, min_periods=lookback).max()
        df["rolling_low"] = df["Low"].rolling(lookback, min_periods=lookback).min()
        df["signal"] = 0
        df.loc[df["Close"] > df["rolling_high"].shift(1), "signal"] = 1
        df.loc[df["Close"] < df["rolling_low"].shift(1), "signal"] = 0
        df["signal"] = _apply_trend_filter(df, df["signal"]).ffill().fillna(0)
        return df


@register_strategy
class BuyHoldStrategy(Strategy):
    name = "buy_hold"
    description = "Buy and hold baseline"

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["signal"] = _apply_trend_filter(df, pd.Series(1.0, index=df.index)).ffill().fillna(0)
        return df


__all__ = [
    "SMARSIStrategy",
    "SMACrossStrategy",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "BreakoutStrategy",
    "BuyHoldStrategy",
]
def _long_trend_mask(data: pd.DataFrame) -> pd.Series:
    price = data["Close"]
    sma50 = sma(price, 50)
    sma200 = sma(price, 200)
    slope50 = sma50.diff()
    return ((price > sma200) & (slope50 > 0)).fillna(False)


def _apply_trend_filter(data: pd.DataFrame, signal: pd.Series) -> pd.Series:
    trend_mask = _long_trend_mask(data)
    filtered = signal.copy()
    filtered = filtered.where(trend_mask, 0)
    return filtered

