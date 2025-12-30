from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

from trader.analytics.metrics import compute_metrics
from trader.analytics.plots import plot_equity, plot_price_with_signals, plot_symbol_returns
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
    output_dir: Path


class BacktestEngine:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.loader = DataLoader()

    def _instantiate_strategy(self) -> Strategy:
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
        )
        per_symbol_results: Dict[str, pd.DataFrame] = {}
        weight = 1 / max(len(data), 1)
        for symbol, frame in data.items():
            signals = strategy.generate_signals(frame)
            per_symbol_results[symbol] = portfolio.backtest_symbol(signals, weight)

        portfolio_result = portfolio.combine(per_symbol_results)
        metrics = compute_metrics(portfolio_result.equity_curve, portfolio_result.trades, self.config.starting_cash)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / timestamp

        write_reports(per_symbol_results, portfolio_result.equity_curve, portfolio_result.trades, self.config.starting_cash, output_dir)
        plot_equity(portfolio_result.equity_curve, output_dir)
        plot_symbol_returns(portfolio_result.per_symbol, output_dir)
        for symbol, frame in per_symbol_results.items():
            plot_price_with_signals(frame, output_dir, symbol)

        return EngineResult(
            equity_curve=portfolio_result.equity_curve,
            trades=portfolio_result.trades,
            metrics=metrics,
            symbol_frames=per_symbol_results,
            output_dir=output_dir,
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
        metrics = compute_metrics(portfolio_result.equity_curve, portfolio_result.trades, self.config.starting_cash)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / f"walkforward_{timestamp}"

        write_reports(per_symbol_results, portfolio_result.equity_curve, portfolio_result.trades, self.config.starting_cash, output_dir)
        plot_equity(portfolio_result.equity_curve, output_dir)
        plot_symbol_returns(portfolio_result.per_symbol, output_dir)
        for symbol, frame in per_symbol_results.items():
            plot_price_with_signals(frame, output_dir, symbol)

        return EngineResult(
            equity_curve=portfolio_result.equity_curve,
            trades=portfolio_result.trades,
            metrics=metrics,
            symbol_frames=per_symbol_results,
            output_dir=output_dir,
        )

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
