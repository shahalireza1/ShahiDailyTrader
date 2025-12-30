from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from trader.strategies.base import Strategy, register_strategy, registry


@dataclass
class EnsembleParameters:
    mode: str = "equal_weight"
    weights: Optional[Dict[str, float]] = None
    threshold: float = 0.5
    vol_lookback: int = 20
    min_weight: float = 0.0


@register_strategy
class EnsembleStrategy(Strategy):
    name = "ensemble"
    description = "Mixture-of-strategies wrapper"

    def __init__(self, strategies: List[Dict[str, object]], **parameters) -> None:
        super().__init__(**parameters)
        if not strategies:
            raise ValueError("EnsembleStrategy requires at least one strategy config")
        self.strategy_configs = strategies
        self.params = EnsembleParameters(**{**EnsembleParameters().__dict__, **parameters})

    def _instantiate(self) -> List[Strategy]:
        instances: List[Strategy] = []
        for cfg in self.strategy_configs:
            name = str(cfg.get("name"))
            params = dict(cfg.get("parameters", {}))
            instances.append(registry.create(name, **params))
        return instances

    def _risk_budget_weights(self, signal_frame: pd.DataFrame) -> pd.DataFrame:
        diffs = signal_frame.diff().fillna(0)
        rolling_vol = diffs.rolling(self.params.vol_lookback).std().replace(0, np.nan)
        inv_vol = 1 / rolling_vol
        inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan)
        weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
        weights = weights.fillna(0)
        return weights

    def _resolve_weights(self, signal_frame: pd.DataFrame) -> pd.DataFrame:
        mode = self.params.mode
        if mode in {"fixed_weights", "fixed"} and self.params.weights:
            raw = pd.Series(self.params.weights)
            aligned = raw.reindex(signal_frame.columns).fillna(0)
            norm = aligned / aligned.abs().sum() if aligned.abs().sum() > 0 else aligned
            weight_row = pd.DataFrame([norm] * len(signal_frame), index=signal_frame.index)
            return weight_row
        if mode in {"risk_budget", "risk_budgeted"}:
            risk_weights = self._risk_budget_weights(signal_frame)
            min_w = max(0.0, float(self.params.min_weight))
            risk_weights = risk_weights.clip(lower=min_w)
            total = risk_weights.sum(axis=1)
            total = total.replace(0, np.nan)
            return risk_weights.div(total, axis=0).fillna(0)
        # default equal weight
        equal = 1 / max(len(signal_frame.columns), 1)
        weight_row = pd.DataFrame(equal, index=signal_frame.index, columns=signal_frame.columns)
        return weight_row

    def _combine_signals(self, signal_frame: pd.DataFrame) -> pd.Series:
        mode = self.params.mode
        weights = self._resolve_weights(signal_frame)
        if mode == "voting":
            thresh = float(self.params.threshold)
            votes = signal_frame.mean(axis=1)
            return (votes >= thresh).astype(float)
        weighted = (signal_frame * weights).sum(axis=1)
        return weighted.clip(-1, 1)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        strategies = self._instantiate()
        signal_cols: Dict[str, pd.Series] = {}
        for strat in strategies:
            signals = strat.generate_signals(data)
            signal_cols[f"signal_{strat.name}"] = signals["signal"].astype(float)
        signals_df = pd.DataFrame(signal_cols)
        signals_df = signals_df.reindex(data.index).fillna(method="ffill").fillna(0)
        weights_df = self._resolve_weights(signals_df)
        combined_signal = self._combine_signals(signals_df)
        df = data.copy()
        for col in signals_df.columns:
            df[col] = signals_df[col]
        for col in weights_df.columns:
            df[f"weight_{col.split('signal_',1)[-1] if col.startswith('signal_') else col}"] = weights_df[col]
        df["signal"] = combined_signal
        df["combined_weight"] = weights_df.sum(axis=1)
        return df
