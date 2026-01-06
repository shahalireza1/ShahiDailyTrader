from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from trader.core.risk import (
    PositionSizingConfig,
    apply_drawdown_stop,
    apply_volatility_target,
    dynamic_exposure_scaler,
    position_sizer,
)


@dataclass
class PortfolioResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    per_symbol: Dict[str, pd.Series]
    positions: pd.DataFrame
    gross_exposure: pd.Series
    portfolio_returns: pd.Series
    turnover: float
    exposure: float
    transaction_costs: pd.Series


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
        eps: float = 1e-6,
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
        self.eps = eps

    def _compute_positions(
        self, executed_signal: pd.Series, returns: pd.Series, per_symbol_fraction: float
    ) -> pd.Series:
        base_position = executed_signal.fillna(0) * per_symbol_fraction
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
        if "signal" not in working and "weight" not in working:
            raise ValueError("Input data must include a 'signal' or 'weight' column")

        target = working["weight"] if "weight" in working else working["signal"]
        target = pd.to_numeric(target, errors="coerce")
        if target.isna().any():
            raise ValueError("Signal series contains non-numeric values")
        if (target.abs() > 1.0).any():
            raise ValueError("Signal and weight values must be within [-1, 1]")

        executed_signal = target.shift(1).fillna(0)
        fraction = position_sizer(working["return"], self.risk_config) * symbol_weight
        position = self._compute_positions(executed_signal, working["return"], fraction)
        position = self._apply_trade_cooldown(position)
        working["executed_signal"] = executed_signal
        working["position"] = position.clip(-self.max_position_per_symbol, self.max_position_per_symbol)

        position_change = working["position"].diff().fillna(working["position"])
        working["strategy_return"] = working["position"].shift(1).fillna(0) * working["return"]
        return working

    def _classify_action(self, prev_weight: float, new_weight: float) -> str:
        if prev_weight == 0 and new_weight != 0:
            return "BUY" if new_weight > 0 else "SELL"
        if new_weight == 0 and prev_weight != 0:
            return "SELL" if prev_weight > 0 else "BUY"
        if prev_weight * new_weight > 0:
            return "ADJUST"
        return "ADJUST"

    def _build_trade_log(
        self,
        positions: pd.DataFrame,
        per_symbol_results: Dict[str, pd.DataFrame],
        equity_curve: pd.Series,
    ) -> pd.DataFrame:
        records: List[Dict[str, float | str | pd.Timestamp]] = []
        if positions.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "ticker",
                    "delta_weight",
                    "prev_weight",
                    "new_weight",
                    "price_used",
                    "notional",
                    "fees",
                    "slippage",
                    "pnl",
                ]
            )

        prev_positions = positions.shift().fillna(0.0)
        equity_by_day = equity_curve.shift().fillna(self.starting_cash)
        fee_rate = self.fee_bps / 10_000
        slip_rate = self.slippage_bps / 10_000

        for ts in positions.index:
            equity_val = float(equity_by_day.loc[ts]) if ts in equity_by_day.index else self.starting_cash
            for symbol in positions.columns:
                prev_weight = float(prev_positions.at[ts, symbol])
                new_weight = float(positions.at[ts, symbol])
                delta_weight = new_weight - prev_weight
                if abs(delta_weight) < self.eps:
                    continue

                price = float(per_symbol_results[symbol].loc[ts, "Close"])
                notional_change = abs(delta_weight) * equity_val
                fees = notional_change * fee_rate
                slippage = notional_change * slip_rate

                records.append(
                    {
                        "date": ts,
                        "ticker": symbol,
                        "delta_weight": delta_weight,
                        "prev_weight": prev_weight,
                        "new_weight": new_weight,
                        "price_used": price,
                        "notional": notional_change,
                        "fees": fees,
                        "slippage": slippage,
                        "pnl": 0.0,
                    }
                )

        return pd.DataFrame.from_records(records)

    def combine(self, per_symbol_results: Dict[str, pd.DataFrame]) -> PortfolioResult:
        returns_df = pd.DataFrame({sym: df["return"] for sym, df in per_symbol_results.items()})
        returns_df.fillna(0, inplace=True)
        positions_df = pd.DataFrame({sym: df["position"] for sym, df in per_symbol_results.items()})
        positions_df.fillna(0, inplace=True)

        base_portfolio_returns = (positions_df.shift().fillna(0) * returns_df).sum(axis=1)
        risk_scaler = dynamic_exposure_scaler(
            base_portfolio_returns,
            drawdown_stop=self.max_drawdown_stop or self.max_drawdown,
            safe_fraction=self.drawdown_safe_fraction,
            rolling_window=self.risk_config.vol_lookback,
        )
        if not risk_scaler.empty:
            positions_df = positions_df.mul(risk_scaler, axis=0)

        if self.max_gross_exposure and self.max_gross_exposure < 1.0:
            gross = positions_df.abs().sum(axis=1)
            scaling = gross.copy()
            scaling[gross > 0] = (self.max_gross_exposure / gross).clip(upper=1.0)
            scaling[gross == 0] = 1.0
            positions_df = positions_df.mul(scaling, axis=0)

        adjusted_results: Dict[str, pd.Series] = {}
        for sym, df in per_symbol_results.items():
            scaled_pos = positions_df[sym].clip(-self.max_position_per_symbol, self.max_position_per_symbol)
            adjusted_results[sym] = scaled_pos.shift().fillna(0) * df["return"]
            per_symbol_results[sym] = df.assign(position=scaled_pos, strategy_return=adjusted_results[sym])

        stop_threshold = self.max_drawdown_stop if self.max_drawdown_stop is not None else self.max_drawdown
        pre_stop_equity = (1 + base_portfolio_returns).cumprod() * self.starting_cash
        active_mask = apply_drawdown_stop(pre_stop_equity, stop_threshold, self.drawdown_safe_fraction)

        effective_positions = positions_df.mul(active_mask, axis=0)
        gross_exposure_series = effective_positions.abs().sum(axis=1)

        equity_values: List[float] = []
        transaction_costs: List[float] = []
        equity = float(self.starting_cash)
        prev_weights = pd.Series(0.0, index=effective_positions.columns)
        transaction_rate = self.transaction_rate
        for idx in effective_positions.index:
            current_weights = effective_positions.loc[idx]
            asset_returns = returns_df.loc[idx]
            weighted_return = float((prev_weights * asset_returns).sum())
            delta_w = current_weights - prev_weights
            cost = 0.0
            if abs(delta_w).sum() > self.eps:
                cost = equity * transaction_rate * abs(delta_w).sum()
            equity = equity * (1 + weighted_return) - cost
            equity_values.append(equity)
            transaction_costs.append(cost)
            prev_weights = current_weights

        equity_curve = pd.Series(equity_values, index=effective_positions.index)
        portfolio_returns = equity_curve.pct_change().fillna(0)
        transaction_cost_series = pd.Series(transaction_costs, index=effective_positions.index)

        trades_df = self._build_trade_log(effective_positions, per_symbol_results, equity_curve)

        if trades_df.empty and (effective_positions.abs() > self.eps).any().any():
            raise ValueError("TRADE_LOG_BROKEN: positions changed but no trades were recorded")

        if trades_df.empty:
            # Ensure that runs with no executed trades remain flat and report zero returns
            equity_curve = pd.Series(self.starting_cash, index=returns_df.index)
            portfolio_returns = pd.Series(0.0, index=returns_df.index)
            gross_exposure_series = pd.Series(0.0, index=returns_df.index)
            turnover = 0.0
            per_symbol_portfolio = returns_df.mul(0.0)
            return PortfolioResult(
                equity_curve=equity_curve,
                trades=trades_df,
                per_symbol={sym: series for sym, series in per_symbol_portfolio.items()},
                positions=effective_positions,
                gross_exposure=gross_exposure_series,
                portfolio_returns=portfolio_returns,
                turnover=turnover,
                exposure=0.0,
                transaction_costs=transaction_cost_series,
            )

        gross_exposure = float(gross_exposure_series.mean()) if not gross_exposure_series.empty else 0.0
        delta_w = effective_positions.diff().fillna(effective_positions)
        turnover = 0.0 if trades_df.empty else float(delta_w.abs().sum().sum() / 2)
        per_symbol_portfolio = effective_positions.shift().fillna(0).mul(returns_df)

        return PortfolioResult(
            equity_curve=equity_curve,
            trades=trades_df,
            per_symbol={sym: series for sym, series in per_symbol_portfolio.items()},
            positions=effective_positions,
            gross_exposure=gross_exposure_series,
            portfolio_returns=portfolio_returns,
            turnover=turnover,
            exposure=gross_exposure,
            transaction_costs=transaction_cost_series,
        )
