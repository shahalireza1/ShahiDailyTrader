from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

from trader.analytics.metrics import compute_metrics
from trader.analytics.plots import (
    plot_drawdown,
    plot_equity,
    plot_equity_vs_benchmark,
    plot_equity_with_drawdown,
    plot_exposure,
    plot_monthly_returns_heatmap,
    plot_price_with_signals,
    plot_rolling_sharpe,
    plot_spy_comparison,
    plot_symbol_returns,
    plot_strategy_contribution,
)
from trader.analytics.reports import generate_html_report, write_reports
from trader.core.diagnostics import PipelineDiagnostics
from trader.core.portfolio import Portfolio
from trader.core.risk import PositionSizingConfig
from trader.data.loaders import DataLoader, DataRequest
from trader.strategies.base import Strategy, registry
from trader.utils.config import Config, config_to_dict


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
    gross_exposure: Optional[pd.Series] = None
    portfolio_returns: Optional[pd.Series] = None
    spy_benchmark: Optional[pd.Series] = None
    rolling_sharpe: Optional[pd.Series] = None
    report_path: Optional[Path] = None
    summary: Optional[Dict[str, Any]] = None


class BacktestEngine:
    def __init__(
        self,
        config: Config,
        enable_plots: bool = False,
        generate_html: bool = False,
        preloaded_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        self.config = config
        self.loader = DataLoader()
        self.enable_plots = enable_plots
        self.generate_html = generate_html
        self.preloaded_data = preloaded_data

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

    def _create_backtest_report(
        self, output_dir: Path, equity_curve: pd.Series, trades: pd.DataFrame, summary: Dict[str, float]
    ) -> Path:
        equity_df = equity_curve.rename("equity").to_frame()
        trades_df = trades.copy()

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "equity.csv").write_text(equity_df.to_csv())
        trades_path = output_dir / "trades.csv"
        trades_df.to_csv(trades_path, index=False)

        config_dict = config_to_dict(self.config)
        (output_dir / "config_used.yaml").write_text(yaml.safe_dump(config_dict, sort_keys=False))
        return generate_html_report(output_dir, summary, equity_df, trades_df, config_dict)

    def _build_run_summary(
        self,
        metrics: Dict[str, float],
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        positions: pd.DataFrame,
        benchmark_curve: pd.Series,
        symbols: Dict[str, pd.DataFrame],
        gross_exposure: pd.Series | float | int | None,
        transaction_costs: pd.Series | None,
        diagnostics: PipelineDiagnostics | None,
    ) -> Dict[str, float | int | str | list | dict]:
        days_with_position = int((positions.abs().sum(axis=1) > 0).sum()) if not positions.empty else 0
        max_abs_position_weight = float(positions.abs().max().max()) if not positions.empty else 0.0
        position_diff = positions.diff().abs().sum(axis=1) if not positions.empty else pd.Series(dtype=float)
        num_position_changes = int(position_diff.gt(1e-9).sum()) if not position_diff.empty else 0
        equity_change_days = int(equity_curve.diff().abs().gt(1e-9).sum()) if not equity_curve.empty else 0
        num_trades = int(metrics.get("num_trades", 0))
        avg_position_size = 0.0
        avg_gross_exposure = 0.0
        avg_hold_days = 0.0
        if not positions.empty:
            active_values = positions.abs().where(positions.abs() > 1e-9).stack()
            avg_position_size = float(active_values.mean()) if not active_values.empty else 0.0
            gross_series = positions.abs().sum(axis=1)
            avg_gross_exposure = float(gross_series.mean()) if not gross_series.empty else 0.0

            hold_lengths: list[int] = []
            for col in positions.columns:
                series = positions[col]
                current = 0
                in_position = False
                for value in series:
                    if abs(value) > 1e-9:
                        current += 1
                        in_position = True
                    elif in_position:
                        hold_lengths.append(current)
                        current = 0
                        in_position = False
                if in_position and current > 0:
                    hold_lengths.append(current)
            if hold_lengths:
                avg_hold_days = float(pd.Series(hold_lengths).mean())

        trades_per_year = 0.0
        if num_trades > 0 and not equity_curve.empty:
            span_days = max((equity_curve.index[-1] - equity_curve.index[0]).days, 1)
            trades_per_year = num_trades / (span_days / 365.25)

        flags: list[str] = []
        if num_position_changes > 0 and num_trades == 0:
            flags.append("TRADE_LOG_BROKEN")
        if days_with_position == 0 and equity_change_days > 0:
            flags.append("EQUITY_NOT_FROM_POSITIONS")

        benchmark_active = not benchmark_curve.empty and bool(benchmark_curve.diff().abs().gt(1e-9).any())
        if days_with_position == 0 and equity_change_days > 0:
            return_source = "benchmark_only"
        elif benchmark_active and days_with_position > 0:
            return_source = "mixed"
        else:
            return_source = "positions_only"

        benchmark_note = ""
        if benchmark_active:
            benchmark_symbols = list(symbols.keys())
            benchmark_focus = "SPY" if "SPY" in benchmark_symbols else (benchmark_symbols[0] if benchmark_symbols else "")
            benchmark_note = (
                f"Benchmark comparison uses buy-and-hold of {benchmark_focus or 'available symbols'}; it does not alter equity"
            )

        # Diagnostics block
        gross_series = gross_exposure if isinstance(gross_exposure, pd.Series) else None
        if gross_series is None:
            gross_series = positions.abs().sum(axis=1) if not positions.empty else pd.Series(dtype=float)
        pct_days_in_cash = float((gross_series < 0.01).mean()) if not gross_series.empty else 0.0
        pct_days_in_market = float((gross_series >= 0.30).mean()) if not gross_series.empty else 0.0
        median_gross = float(gross_series.median()) if not gross_series.empty else 0.0
        p90_gross = float(gross_series.quantile(0.9)) if not gross_series.empty else 0.0

        per_symbol_signal: Dict[str, Dict[str, float]] = {}
        per_symbol_weights: Dict[str, float] = {}
        overall_signals: list[pd.Series] = []
        overall_weights: list[pd.Series] = []
        for sym, frame in symbols.items():
            signal_series = frame.get("signal", frame.get("raw_signal", pd.Series(dtype=float))).fillna(0)
            per_symbol_signal[sym] = {
                "avg_abs_signal": float(signal_series.abs().mean()) if not signal_series.empty else 0.0,
                "pct_nonzero_signal_days": float((signal_series.abs() > 1e-9).mean()) if not signal_series.empty else 0.0,
            }
            overall_signals.append(signal_series)

            final_positions = frame.get("final_position", frame.get("position", pd.Series(dtype=float))).fillna(0)
            active_mask = final_positions.abs() > 1e-9
            avg_active_weight = float(final_positions[active_mask].abs().mean()) if active_mask.any() else 0.0
            per_symbol_weights[sym] = avg_active_weight
            overall_weights.append(final_positions.abs())

        combined_signals = pd.concat(overall_signals) if overall_signals else pd.Series(dtype=float)
        combined_weights = pd.concat(overall_weights) if overall_weights else pd.Series(dtype=float)
        active_overall_weights = combined_weights[combined_weights > 1e-9] if not combined_weights.empty else pd.Series(dtype=float)
        overall_weight_active = float(active_overall_weights.mean()) if not active_overall_weights.empty else 0.0

        diagnostics_block = {
            "pct_days_in_cash": pct_days_in_cash,
            "pct_days_in_market": pct_days_in_market,
            "avg_gross_exposure": float(gross_series.mean()) if not gross_series.empty else 0.0,
            "median_gross_exposure": median_gross,
            "p90_gross_exposure": p90_gross,
            "signal_activity": {
                "overall": {
                    "avg_abs_signal": float(combined_signals.abs().mean()) if not combined_signals.empty else 0.0,
                    "pct_nonzero_signal_days": float((combined_signals.abs() > 1e-9).mean())
                    if not combined_signals.empty
                    else 0.0,
                },
                "per_symbol": per_symbol_signal,
            },
            "avg_target_weight_when_active": {
                "overall": overall_weight_active,
                "per_symbol": per_symbol_weights,
            },
            "block_reasons": diagnostics.export() if diagnostics else {},
            "transaction_costs_total": float(transaction_costs.sum()) if transaction_costs is not None else 0.0,
            "transaction_costs_bps_of_equity": (
                float(transaction_costs.sum()) / self.config.starting_cash * 10_000
                if transaction_costs is not None and self.config.starting_cash
                else 0.0
            ),
        }

        summary = {
            "cagr": float(metrics.get("cagr", 0.0)),
            "total_return": float(metrics.get("total_return", 0.0)),
            "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
            "sharpe": float(metrics.get("sharpe", 0.0)),
            "num_trades": num_trades,
            "win_rate": float(metrics.get("win_rate", 0.0)),
            "days_with_position": days_with_position,
            "max_abs_position_weight": max_abs_position_weight,
            "num_position_changes": num_position_changes,
            "equity_change_days": equity_change_days,
            "avg_position_size": avg_position_size,
            "avg_gross_exposure": avg_gross_exposure,
            "avg_hold_days": avg_hold_days,
            "trades_per_year": trades_per_year,
            "flags": flags,
            "return_source": return_source,
            "diagnostics": diagnostics_block,
        }
        if benchmark_note:
            summary["benchmark_details"] = benchmark_note
        return summary

    def _write_summary(self, output_dir: Path, summary: Dict[str, Any]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    def _run_single_backtest(self) -> EngineResult:
        strategy = self._instantiate_strategy()
        data = self.loader.fetch_many(
            (DataRequest(symbol=s, start=self.config.start, end=self.config.end) for s in self.config.symbols),
            preloaded_data=self.preloaded_data,
        )
        risk_cfg = PositionSizingConfig(
            mode=self.config.risk.position_mode,
            fraction=self.config.risk.position_fraction,
            kelly_safety=self.config.risk.kelly_safety,
            target_volatility=self.config.risk.target_volatility,
            vol_lookback=self.config.risk.vol_lookback,
        )
        diagnostics = PipelineDiagnostics()
        portfolio = Portfolio(
            starting_cash=self.config.starting_cash,
            fee_bps=self.config.fees_bps,
            slippage_bps=self.config.slippage_bps,
            risk_config=risk_cfg,
            max_drawdown=self.config.risk.max_drawdown,
            max_drawdown_stop=self.config.risk.max_drawdown_stop,
            drawdown_safe_fraction=self.config.risk.drawdown_safe_fraction,
            max_gross_exposure=self.config.risk.max_gross_exposure,
            target_gross_exposure=self.config.risk.target_gross_exposure,
            max_position_per_symbol=self.config.risk.max_position_per_symbol,
            trade_cooldown_days=self.config.risk.trade_cooldown_days,
            min_target_weight=self.config.risk.min_active_weight,
            rebalance_band=self.config.risk.rebalance_band,
            signal_frequency=self.config.risk.signal_frequency,
            signal_persistence_days=self.config.risk.signal_persistence_days,
            min_hold_days=self.config.risk.min_hold_days,
            diagnostics=diagnostics,
        )
        per_symbol_results: Dict[str, pd.DataFrame] = {}
        weight = 1 / max(len(data), 1)
        for symbol, frame in data.items():
            signals = strategy.generate_signals(frame)
            per_symbol_results[symbol] = portfolio.backtest_symbol(signals, weight)

        portfolio_result = portfolio.combine(per_symbol_results)
        positions_df = portfolio_result.positions
        gross_exposure = portfolio_result.exposure if portfolio_result.exposure is not None else 0.0
        turnover = portfolio_result.turnover if portfolio_result.turnover is not None else 0.0

        benchmark_curve = self._build_benchmark(per_symbol_results)
        spy_benchmark = self._spy_buy_and_hold(per_symbol_results)
        rolling_sharpe = self._rolling_sharpe(portfolio_result.portfolio_returns)
        metrics = compute_metrics(
            portfolio_result.equity_curve,
            portfolio_result.trades,
            self.config.starting_cash,
            exposure=gross_exposure,
            turnover=turnover,
            transaction_costs=portfolio_result.transaction_costs,
        )
        attribution = self._strategy_attribution(per_symbol_results)
        monthly_returns = self._monthly_returns(portfolio_result.equity_curve)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / timestamp

        summary = self._build_run_summary(
            metrics,
            portfolio_result.equity_curve,
            portfolio_result.trades,
            positions_df,
            benchmark_curve,
            per_symbol_results,
            portfolio_result.gross_exposure,
            portfolio_result.transaction_costs,
            diagnostics,
        )
        self._write_summary(output_dir, summary)

        write_reports(
            per_symbol_results,
            portfolio_result.equity_curve,
            portfolio_result.trades,
            self.config.starting_cash,
            output_dir,
            metrics,
        )
        report_path = None
        if self.generate_html:
            report_path = self._create_backtest_report(output_dir, portfolio_result.equity_curve, portfolio_result.trades, summary)
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
            if portfolio_result.gross_exposure is not None:
                plot_exposure(portfolio_result.gross_exposure, plots_dir)
            if rolling_sharpe is not None:
                plot_rolling_sharpe(rolling_sharpe, plots_dir)
            if attribution is not None:
                plot_strategy_contribution(attribution, plots_dir)
            if spy_benchmark is not None:
                plot_spy_comparison(portfolio_result.equity_curve, spy_benchmark, plots_dir)
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
            gross_exposure=portfolio_result.gross_exposure,
            portfolio_returns=portfolio_result.portfolio_returns,
            spy_benchmark=spy_benchmark,
            rolling_sharpe=rolling_sharpe,
            report_path=report_path,
            summary=summary,
        )

    def _run_walkforward(self) -> EngineResult:
        strategy = self._instantiate_strategy()
        data = self.loader.fetch_many(
            (DataRequest(symbol=s, start=self.config.start, end=self.config.end) for s in self.config.symbols),
            preloaded_data=self.preloaded_data,
        )
        risk_cfg = PositionSizingConfig(
            mode=self.config.risk.position_mode,
            fraction=self.config.risk.position_fraction,
            kelly_safety=self.config.risk.kelly_safety,
            target_volatility=self.config.risk.target_volatility,
            vol_lookback=self.config.risk.vol_lookback,
        )
        diagnostics = PipelineDiagnostics()
        portfolio = Portfolio(
            starting_cash=self.config.starting_cash,
            fee_bps=self.config.fees_bps,
            slippage_bps=self.config.slippage_bps,
            risk_config=risk_cfg,
            max_drawdown=self.config.risk.max_drawdown,
            max_drawdown_stop=self.config.risk.max_drawdown_stop,
            drawdown_safe_fraction=self.config.risk.drawdown_safe_fraction,
            max_gross_exposure=self.config.risk.max_gross_exposure,
            target_gross_exposure=self.config.risk.target_gross_exposure,
            max_position_per_symbol=self.config.risk.max_position_per_symbol,
            trade_cooldown_days=self.config.risk.trade_cooldown_days,
            min_target_weight=self.config.risk.min_active_weight,
            rebalance_band=self.config.risk.rebalance_band,
            signal_frequency=self.config.risk.signal_frequency,
            signal_persistence_days=self.config.risk.signal_persistence_days,
            min_hold_days=self.config.risk.min_hold_days,
            diagnostics=diagnostics,
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
        positions_df = portfolio_result.positions
        gross_exposure = portfolio_result.exposure if portfolio_result.exposure is not None else 0.0
        turnover = portfolio_result.turnover if portfolio_result.turnover is not None else 0.0

        benchmark_curve = self._build_benchmark(per_symbol_results)
        spy_benchmark = self._spy_buy_and_hold(per_symbol_results)
        rolling_sharpe = self._rolling_sharpe(portfolio_result.portfolio_returns)
        metrics = compute_metrics(
            portfolio_result.equity_curve,
            portfolio_result.trades,
            self.config.starting_cash,
            exposure=gross_exposure,
            turnover=turnover,
            transaction_costs=portfolio_result.transaction_costs,
        )
        attribution = self._strategy_attribution(per_symbol_results)
        monthly_returns = self._monthly_returns(portfolio_result.equity_curve)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.config.output_dir) / f"walkforward_{timestamp}"

        summary = self._build_run_summary(
            metrics,
            portfolio_result.equity_curve,
            portfolio_result.trades,
            positions_df,
            benchmark_curve,
            per_symbol_results,
            portfolio_result.gross_exposure,
            portfolio_result.transaction_costs,
            diagnostics,
        )
        self._write_summary(output_dir, summary)

        write_reports(
            per_symbol_results,
            portfolio_result.equity_curve,
            portfolio_result.trades,
            self.config.starting_cash,
            output_dir,
            metrics,
        )
        report_path = None
        if self.generate_html:
            report_path = self._create_backtest_report(output_dir, portfolio_result.equity_curve, portfolio_result.trades, summary)
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
            if portfolio_result.gross_exposure is not None:
                plot_exposure(portfolio_result.gross_exposure, plots_dir)
            if rolling_sharpe is not None:
                plot_rolling_sharpe(rolling_sharpe, plots_dir)
            if attribution is not None:
                plot_strategy_contribution(attribution, plots_dir)
            if spy_benchmark is not None:
                plot_spy_comparison(portfolio_result.equity_curve, spy_benchmark, plots_dir)
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
            gross_exposure=portfolio_result.gross_exposure,
            portfolio_returns=portfolio_result.portfolio_returns,
            spy_benchmark=spy_benchmark,
            rolling_sharpe=rolling_sharpe,
            report_path=report_path,
            summary=summary,
        )

    def _build_benchmark(self, per_symbol_results: Dict[str, pd.DataFrame]) -> pd.Series:
        returns_df = pd.DataFrame({sym: df["return"] for sym, df in per_symbol_results.items()})
        returns_df.fillna(0, inplace=True)
        benchmark_returns = returns_df.mean(axis=1)
        return (1 + benchmark_returns).cumprod() * self.config.starting_cash

    def _spy_buy_and_hold(self, per_symbol_results: Dict[str, pd.DataFrame]) -> pd.Series:
        if not per_symbol_results:
            return pd.Series(dtype=float)
        if "SPY" in per_symbol_results:
            spy = per_symbol_results["SPY"]
        else:
            spy = next(iter(per_symbol_results.values()))
        base_price = spy["Close"].iloc[0]
        returns = spy["Close"] / base_price
        return returns * self.config.starting_cash

    def _rolling_sharpe(self, portfolio_returns: pd.Series, window: int = 126) -> pd.Series:
        if portfolio_returns is None or portfolio_returns.empty:
            return pd.Series(dtype=float)
        sharpe = portfolio_returns.rolling(window).apply(
            lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else 0.0,
            raw=False,
        )
        return sharpe

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
