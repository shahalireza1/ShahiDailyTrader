from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from trader.core.risk import PositionSizingConfig, apply_drawdown_stop, apply_volatility_target, position_sizer


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl: float


@dataclass
class PortfolioResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    per_symbol: Dict[str, pd.Series]


class Portfolio:
    def __init__(
        self,
        starting_cash: float,
        fee_bps: float,
        slippage_bps: float,
        risk_config: PositionSizingConfig,
        max_drawdown: float,
        max_drawdown_stop: float | None = None,
        drawdown_safe_fraction: float = 0.0,
        max_gross_exposure: float = 1.0,
        max_position_per_symbol: float = 1.0,
        trade_cooldown_days: int = 0,
    ) -> None:
        self.starting_cash = starting_cash
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self.risk_config = risk_config
        self.max_drawdown = max_drawdown
        self.max_drawdown_stop = max_drawdown_stop
        self.drawdown_safe_fraction = drawdown_safe_fraction
        self.max_gross_exposure = max_gross_exposure
        self.max_position_per_symbol = max_position_per_symbol
        self.trade_cooldown_days = trade_cooldown_days
        self.transaction_rate = (self.fee_bps + self.slippage_bps) / 10_000

    def _generate_trades(self, df: pd.DataFrame, symbol: str) -> List[Trade]:
        trades: List[Trade] = []
        prev_signal = 0
        entry_price: Optional[float] = None
        entry_date: Optional[pd.Timestamp] = None
        for ts, row in df.iterrows():
            signal = int(round(row["position"]))
            price = float(row["Close"])
            if prev_signal == 0 and signal > 0:
                entry_price = price
                entry_date = ts
            elif prev_signal > 0 and signal == 0 and entry_price is not None:
                pnl = (price - entry_price) / entry_price
                trades.append(
                    Trade(
                        symbol=symbol,
                        entry_date=entry_date or ts,
                        exit_date=ts,
                        entry_price=entry_price,
                        exit_price=price,
                        pnl=pnl,
                    )
                )
                entry_price = None
                entry_date = None
            prev_signal = signal

        if prev_signal > 0 and entry_price is not None:
            last_price = float(df["Close"].iloc[-1])
            last_ts = df.index[-1]
            pnl = (last_price - entry_price) / entry_price
            trades.append(
                Trade(
                    symbol=symbol,
                    entry_date=entry_date or last_ts,
                    exit_date=last_ts,
                    entry_price=entry_price,
                    exit_price=last_price,
                    pnl=pnl,
                )
            )
        return trades

    def _compute_positions(
        self, df: pd.DataFrame, returns: pd.Series, per_symbol_fraction: float
    ) -> pd.Series:
        base_position = df["signal"].shift(1).fillna(0) * per_symbol_fraction
        vol_adj = apply_volatility_target(
            base_position, returns, self.risk_config.target_volatility, self.risk_config.vol_lookback
        )
        return vol_adj.clip(-1, 1)

    def _apply_trade_cooldown(self, position: pd.Series) -> pd.Series:
        if self.trade_cooldown_days <= 0:
            return position
        cooled = position.copy()
        last_trade_idx: Optional[pd.Timestamp] = None
        last_position = 0.0
        for idx, value in position.items():
            if last_trade_idx is None:
                if value != 0:
                    last_trade_idx = idx
                    last_position = value
                cooled.loc[idx] = value
                continue
            days_since = (idx - last_trade_idx).days
            if abs(value - last_position) > 1e-9 and days_since <= self.trade_cooldown_days:
                cooled.loc[idx] = last_position
            else:
                cooled.loc[idx] = value
                if abs(value - last_position) > 1e-9:
                    last_trade_idx = idx
                    last_position = value
        return cooled

    def backtest_symbol(self, df: pd.DataFrame, symbol_weight: float) -> pd.DataFrame:
        working = df.copy()
        working.sort_index(inplace=True)
        working["return"] = working["Close"].pct_change().fillna(0)
        fraction = position_sizer(working["return"], self.risk_config) * symbol_weight
        position = self._compute_positions(working, working["return"], fraction)
        position = self._apply_trade_cooldown(position)
        working["position"] = position.clip(-self.max_position_per_symbol, self.max_position_per_symbol)

        position_change = working["position"].diff().fillna(working["position"])
        transaction_cost = abs(position_change) * self.transaction_rate
        working["strategy_return"] = working["position"] * working["return"] - transaction_cost
        return working

    def combine(self, per_symbol_results: Dict[str, pd.DataFrame]) -> PortfolioResult:
        returns_df = pd.DataFrame({sym: df["strategy_return"] for sym, df in per_symbol_results.items()})
        returns_df.fillna(0, inplace=True)
        positions_df = pd.DataFrame({sym: df["position"] for sym, df in per_symbol_results.items()})
        positions_df.fillna(0, inplace=True)

        if self.max_gross_exposure and self.max_gross_exposure < 1.0:
            gross = positions_df.abs().sum(axis=1)
            scaling = gross.copy()
            scaling[gross > 0] = (self.max_gross_exposure / gross).clip(upper=1.0)
            scaling[gross == 0] = 1.0
            positions_df = positions_df.mul(scaling, axis=0)

        adjusted_results: Dict[str, pd.Series] = {}
        for sym, df in per_symbol_results.items():
            scaled_pos = positions_df[sym].clip(-self.max_position_per_symbol, self.max_position_per_symbol)
            pos_change = scaled_pos.diff().fillna(scaled_pos)
            cost = abs(pos_change) * self.transaction_rate
            adjusted_results[sym] = scaled_pos * df["return"] - cost
            per_symbol_results[sym] = df.assign(position=scaled_pos, strategy_return=adjusted_results[sym])

        portfolio_returns = pd.DataFrame(adjusted_results).mean(axis=1)

        equity_curve = (1 + portfolio_returns).cumprod() * self.starting_cash
        stop_threshold = self.max_drawdown_stop if self.max_drawdown_stop is not None else self.max_drawdown
        active_mask = apply_drawdown_stop(equity_curve, stop_threshold, self.drawdown_safe_fraction)
        equity_curve = (1 + portfolio_returns * active_mask).cumprod() * self.starting_cash

        all_trades: List[Trade] = []
        for symbol, df in per_symbol_results.items():
            trades = self._generate_trades(df.assign(position=df["position"] * active_mask), symbol)
            all_trades.extend(trades)

        trades_df = pd.DataFrame([t.__dict__ for t in all_trades])
        return PortfolioResult(
            equity_curve=equity_curve,
            trades=trades_df,
            per_symbol={sym: df["strategy_return"] for sym, df in per_symbol_results.items()},
        )
