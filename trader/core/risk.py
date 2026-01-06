from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

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
        soft_mask = active + (1 - active) * safe_fraction
        # soften further when the drawdown deepens beyond the stop
        scaled = 1 + drawdown.clip(upper=-abs(max_drawdown)) / abs(max_drawdown)
        scaled = scaled.clip(lower=safe_fraction, upper=1.0)
        return (soft_mask * 0.5) + (scaled * 0.5)
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


def dynamic_exposure_scaler(
    returns: pd.Series,
    drawdown_stop: Optional[float],
    safe_fraction: float,
    rolling_window: int = 60,
) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)

    window = max(rolling_window, 20)
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = returns.rolling(window).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else 0.0,
        raw=False,
    )

    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    drawdown_accel = drawdown.diff().fillna(0)

    base_scale = pd.Series(0.65, index=returns.index)

    dd_tailwind = ((0.05 + drawdown).clip(lower=0, upper=0.05) / 0.05) * 0.2
    sharpe_tailwind = ((rolling_sharpe - 0.5).clip(lower=0) / 1.5).clip(upper=0.4) * 0.35
    scale = base_scale + dd_tailwind + sharpe_tailwind

    vol_penalty = ((rolling_vol - 0.2).clip(lower=0) / 0.4).clip(upper=0.4) * 0.3
    accel_penalty = (drawdown_accel.clip(lower=0) / 0.02).clip(upper=0.5) * 0.35
    scale -= (vol_penalty + accel_penalty).fillna(0)

    if drawdown_stop:
        stress = (-drawdown) / abs(drawdown_stop)
        stress = stress.clip(lower=0)
        scale = scale * (1 - stress.clip(upper=1)) + safe_fraction * stress.clip(upper=1)

    return scale.clip(lower=max(safe_fraction, 0.25), upper=1.05).fillna(
        max(safe_fraction, 0.25)
    )


def apply_volatility_target(position: pd.Series, returns: pd.Series, target_vol: Optional[float], lookback: int) -> pd.Series:
    if target_vol is None or target_vol <= 0:
        return position
    rolling_vol = returns.rolling(lookback).std() * np.sqrt(252)
    scale = target_vol / rolling_vol.replace(0, np.nan)
    scale = scale.clip(upper=5).fillna(0)
    return (position * scale).clip(-1, 1)
