from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from trader.analytics.metrics import compute_metrics
from trader.analytics.plots import (
    plot_drawdown,
    plot_equity,
    plot_equity_vs_benchmark,
    plot_equity_with_drawdown,
    plot_monthly_returns_heatmap,
    plot_price_with_signals,
    plot_symbol_returns,
)
from trader.analytics.reports import write_reports
from trader.core.portfolio import Portfolio
from trader.core.risk import PositionSizingConfig
from trader.data.loaders import DataLoader, DataRequest
from trader.strategies.base import Strategy, registry
from trader.utils.config import Config


@dataclass
class EngineResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    symbol_frames: Dict[str, pd.DataFrame]
    benchmark_curve: pd.Series
    monthly_returns: pd.DataFrame
    output_dir: Path
    plots_dir: Optional[Path] = None
    strategy_attribution: Optional[pd.DataFrame] = None


class BacktestEngine:
    def __init__(self, config: Config, enable_plots: bool = False) -> None:
        self.config = config
        self.loader = DataLoader()
        self.enable_plots = enable_plots

    def _instantiate_strategy(self) -> Strategy:
        strategies_payload = [
            {"name": cfg.name, "parameters": cfg.parameters} for cfg in (self.config.strategies or [self.config.strategy])
        ]
        if (
            self.config.strategy.name == "ensemble"
            or len(self.config.strategies) > 1
            or bool(self.config.ensemble)
        ):
            ensemble_params = dict(self.config.ensemble)
            if self.config.strategy.name == "ensemble":
                ensemble_params.update(self.config.strategy.parameters)
            return registry.create("ensemble", strategies=strategies_payload, **ensemble_params)
        return registry.create(self.config.strategy.name, **self.config.strategy.parameters)

    def _run_single_backtest(self) -> EngineResult:
        strategy = self._instantiate_strategy()
        data = self.loader.fetch_many(
            DataRequest(symbol=s, start=self.config.start, end=self.config.end) for s in self.config.symbols
        )
        risk_cfg = PositionSizingConfig(
            mode=self.config.risk.position_mode,
            fraction=self.config.risk.position_fraction,
            kelly_safety=self.config.risk.kelly_safety,
            target_volatility=self.config.risk.target_volatility,
            vol_lookback=self.config.risk.vol_lookback,
        )
        portfolio = Portfolio(
            starting_cash=self.config.starting_cash,
            fee_bps=self.config.fees_bps,
            slippage_bps=self.config.slippage_bps,
            risk_config=risk_cfg,
            max_drawdown=self.config.risk.max_drawdown,
            max_drawdown_stop=self.config.risk.max_drawdown_stop,
            drawdown_safe_fraction=self.config.risk.drawdown_safe_fraction,
            max_gross_exposure=self.config.risk.max_gross_exposure,
            max_position_per_symbol=self.config.risk.max_position_per_symbol,
            trade_cooldown_days=self.config.risk.trade_cooldown_days,
        )
        per_symbol_results: Dict[str, pd.DataFrame] = {}
        weight = 1 / max(len(data), 1)
        for symbol, frame in data.items():
            signals = strategy.generate_signals(frame)
            per_symbol_results[symbol] = portfolio.backtest_symbol(signals, weight)

        portfolio_result = portfolio.combine(per_symbol_results)
        positions_df = pd.DataFrame({sym: df["position"] for sym, df in per_symbol_results.items()})
        gross_exposure = positions_df.abs().sum(axis=1).mean() if not positions_df.empty else 0.0
        turnover = positions_df.diff().abs().sum(axis=1).mean() if not positions_df.empty else 0.0

        benchmark_curve = self._build_benchmark(per_symbol_results)
        metrics = compute_metrics(
            portfolio_result.equity_curve,
            portfolio_result.trades,
            self.config.starting_cash,
            exposure=gross_exposure,
            turnover=turnover,
        )
        attribution = self._strategy_attribution(per_symbol_results)
        monthly_returns = self._monthly_returns(portfolio_result.equity_curve)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / timestamp

        write_reports(
            per_symbol_results,
            portfolio_result.equity_curve,
            portfolio_result.trades,
            self.config.starting_cash,
            output_dir,
            metrics,
        )
        if not attribution.empty:
            attribution.to_csv(output_dir / "strategy_attribution.csv")
        plots_dir = None
        if self.enable_plots:
            plots_dir = output_dir / "plots"
            plot_equity(portfolio_result.equity_curve, plots_dir)
            plot_equity_vs_benchmark(portfolio_result.equity_curve, benchmark_curve, plots_dir)
            plot_equity_with_drawdown(portfolio_result.equity_curve, benchmark_curve, plots_dir)
            plot_drawdown(portfolio_result.equity_curve, plots_dir)
            plot_monthly_returns_heatmap(monthly_returns * 100, plots_dir)
            plot_symbol_returns(portfolio_result.per_symbol, plots_dir)
            for symbol, frame in per_symbol_results.items():
                plot_price_with_signals(frame, plots_dir, symbol)

        return EngineResult(
            equity_curve=portfolio_result.equity_curve,
            trades=portfolio_result.trades,
            metrics=metrics,
            symbol_frames=per_symbol_results,
            benchmark_curve=benchmark_curve,
            monthly_returns=monthly_returns,
            output_dir=output_dir,
            plots_dir=plots_dir,
            strategy_attribution=attribution,
        )

    def _run_walkforward(self) -> EngineResult:
        strategy = self._instantiate_strategy()
        data = self.loader.fetch_many(
            DataRequest(symbol=s, start=self.config.start, end=self.config.end) for s in self.config.symbols
        )
        risk_cfg = PositionSizingConfig(
            mode=self.config.risk.position_mode,
            fraction=self.config.risk.position_fraction,
            kelly_safety=self.config.risk.kelly_safety,
            target_volatility=self.config.risk.target_volatility,
            vol_lookback=self.config.risk.vol_lookback,
        )
        portfolio = Portfolio(
            starting_cash=self.config.starting_cash,
            fee_bps=self.config.fees_bps,
            slippage_bps=self.config.slippage_bps,
            risk_config=risk_cfg,
            max_drawdown=self.config.risk.max_drawdown,
            max_drawdown_stop=self.config.risk.max_drawdown_stop,
            drawdown_safe_fraction=self.config.risk.drawdown_safe_fraction,
            max_gross_exposure=self.config.risk.max_gross_exposure,
            max_position_per_symbol=self.config.risk.max_position_per_symbol,
            trade_cooldown_days=self.config.risk.trade_cooldown_days,
        )

        per_symbol_results: Dict[str, pd.DataFrame] = {}
        for symbol, frame in data.items():
            segments = []
            total_len = len(frame)
            start_idx = 0
            while start_idx < total_len:
                train_end = start_idx + self.config.walkforward.train_window
                test_end = train_end + self.config.walkforward.test_window
                test_slice = frame.iloc[train_end:test_end]
                if test_slice.empty:
                    break
                signals = strategy.generate_signals(test_slice)
                segments.append(signals)
                start_idx += self.config.walkforward.step
            if segments:
                stitched = pd.concat(segments).sort_index()
            else:
                stitched = strategy.generate_signals(frame)
            per_symbol_results[symbol] = portfolio.backtest_symbol(stitched, 1 / max(len(data), 1))

        portfolio_result = portfolio.combine(per_symbol_results)
        positions_df = pd.DataFrame({sym: df["position"] for sym, df in per_symbol_results.items()})
        gross_exposure = positions_df.abs().sum(axis=1).mean() if not positions_df.empty else 0.0
        turnover = positions_df.diff().abs().sum(axis=1).mean() if not positions_df.empty else 0.0

        benchmark_curve = self._build_benchmark(per_symbol_results)
        metrics = compute_metrics(
            portfolio_result.equity_curve,
            portfolio_result.trades,
            self.config.starting_cash,
            exposure=gross_exposure,
            turnover=turnover,
        )
        attribution = self._strategy_attribution(per_symbol_results)
        monthly_returns = self._monthly_returns(portfolio_result.equity_curve)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / f"walkforward_{timestamp}"

        write_reports(
            per_symbol_results,
            portfolio_result.equity_curve,
            portfolio_result.trades,
            self.config.starting_cash,
            output_dir,
            metrics,
        )
        if not attribution.empty:
            attribution.to_csv(output_dir / "strategy_attribution.csv")
        plots_dir = None
        if self.enable_plots:
            plots_dir = output_dir / "plots"
            plot_equity(portfolio_result.equity_curve, plots_dir)
            plot_equity_vs_benchmark(portfolio_result.equity_curve, benchmark_curve, plots_dir)
            plot_equity_with_drawdown(portfolio_result.equity_curve, benchmark_curve, plots_dir)
            plot_drawdown(portfolio_result.equity_curve, plots_dir)
            plot_monthly_returns_heatmap(monthly_returns * 100, plots_dir)
            plot_symbol_returns(portfolio_result.per_symbol, plots_dir)
            for symbol, frame in per_symbol_results.items():
                plot_price_with_signals(frame, plots_dir, symbol)

        return EngineResult(
            equity_curve=portfolio_result.equity_curve,
            trades=portfolio_result.trades,
            metrics=metrics,
            symbol_frames=per_symbol_results,
            benchmark_curve=benchmark_curve,
            monthly_returns=monthly_returns,
            output_dir=output_dir,
            plots_dir=plots_dir,
            strategy_attribution=attribution,
        )

    def _build_benchmark(self, per_symbol_results: Dict[str, pd.DataFrame]) -> pd.Series:
        returns_df = pd.DataFrame({sym: df["return"] for sym, df in per_symbol_results.items()})
        returns_df.fillna(0, inplace=True)
        benchmark_returns = returns_df.mean(axis=1)
        return (1 + benchmark_returns).cumprod() * self.config.starting_cash

    def _monthly_returns(self, equity_curve: pd.Series) -> pd.DataFrame:
        daily_returns = equity_curve.pct_change().dropna()
        monthly = (1 + daily_returns).resample("M").prod() - 1
        monthly_df = monthly.to_frame("return")
        monthly_df["Year"] = monthly_df.index.year
        monthly_df["Month"] = monthly_df.index.strftime("%b")
        pivot = monthly_df.pivot(index="Year", columns="Month", values="return").fillna(0)
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        pivot = pivot.reindex(columns=month_order, fill_value=0)
        return pivot

    def run(self) -> EngineResult:
        mode = self.config.mode
        if mode == "backtest":
            return self._run_single_backtest()
        if mode == "walkforward":
            return self._run_walkforward()
        if mode == "paper":
            # Placeholder identical to backtest for now
            return self._run_single_backtest()
        raise ValueError(f"Unsupported mode: {mode}")

    def _strategy_attribution(self, per_symbol_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        records = []
        for symbol, df in per_symbol_results.items():
            combined = df.get("signal", df.get("position", pd.Series(dtype=float)))
            for col in df.columns:
                if col.startswith("signal_"):
                    name = col.replace("signal_", "")
                    corr = combined.corr(df[col]) if len(combined) and df[col].std() != 0 else 0.0
                    overlap = float((combined * df[col]).mean()) if len(combined) else 0.0
                    records.append({"strategy": name, "symbol": symbol, "overlap": overlap, "correlation": corr})
        if not records:
            return pd.DataFrame()
        grouped = pd.DataFrame(records).groupby("strategy").agg({"overlap": "mean", "correlation": "mean"})
        return grouped.sort_values("overlap", ascending=False)
