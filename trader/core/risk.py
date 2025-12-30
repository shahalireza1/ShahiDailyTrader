from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PositionSizingConfig:
    mode: str = "fixed_fraction"
    fraction: float = 1.0
    kelly_safety: float = 0.5
    target_volatility: Optional[float] = None
    vol_lookback: int = 20


def apply_drawdown_stop(
    equity_curve: pd.Series, max_drawdown: float, safe_fraction: float = 0.0
) -> pd.Series:
    if equity_curve.empty or max_drawdown is None or max_drawdown <= 0:
        return pd.Series(1.0, index=equity_curve.index)
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    halted = drawdown < -abs(max_drawdown)
    active = (~halted).astype(float)
    # once halted, stay halted
    active = active.cummin()
    if safe_fraction > 0:
        return active + (1 - active) * safe_fraction
    return active


def _kelly_fraction(returns: pd.Series) -> float:
    wins = returns[returns > 0]
    losses = -returns[returns < 0]
    if wins.empty or losses.empty:
        return 0.0
    win_rate = len(wins) / len(returns)
    win_loss_ratio = wins.mean() / losses.mean()
    kelly = win_rate - (1 - win_rate) / win_loss_ratio
    return max(0.0, min(1.0, kelly))


def position_sizer(returns: pd.Series, config: PositionSizingConfig) -> float:
    mode = config.mode
    if mode == "fixed_fraction":
        return max(0.0, min(1.0, config.fraction))
    if mode == "kelly_lite":
        kelly = _kelly_fraction(returns)
        return max(0.0, min(1.0, kelly * config.kelly_safety))
    return 1.0


def apply_volatility_target(position: pd.Series, returns: pd.Series, target_vol: Optional[float], lookback: int) -> pd.Series:
    if target_vol is None or target_vol <= 0:
        return position
    rolling_vol = returns.rolling(lookback).std() * np.sqrt(252)
    scale = target_vol / rolling_vol.replace(0, np.nan)
    scale = scale.clip(upper=5).fillna(0)
    return (position * scale).clip(-1, 1)
