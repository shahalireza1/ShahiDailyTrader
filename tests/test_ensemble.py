import pandas as pd

from trader.analytics.reports import build_html_report
from trader.core.engine import BacktestEngine
from trader.utils.config import Config, RiskConfig, StrategyConfig


def test_ensemble_respects_exposure_and_builds_report(tmp_path):
    base_output = tmp_path / "outputs"
    config = Config()
    config.symbols = ["SPY", "QQQ"]
    config.start = "2022-01-01"
    config.end = "2022-06-30"
    config.output_dir = base_output
    config.strategies = [
        StrategyConfig(name="sma_cross", parameters={"fast": 15, "slow": 40}),
        StrategyConfig(name="mean_reversion", parameters={"lookback": 8, "entry_z": -1.0, "exit_z": 0.0}),
    ]
    config.strategy = config.strategies[0]
    config.ensemble = {"mode": "equal_weight"}
    config.risk = RiskConfig(
        max_drawdown=0.25,
        max_gross_exposure=0.35,
        max_position_per_symbol=0.02,
        vol_target_annual=0.1,
    )

    engine = BacktestEngine(config, enable_plots=True)
    dates = pd.date_range("2022-01-03", periods=60, freq="B")
    price_series = pd.Series(100 + pd.RangeIndex(len(dates)), index=dates)
    template = pd.DataFrame(
        {
            "Open": price_series,
            "High": price_series * 1.01,
            "Low": price_series * 0.99,
            "Close": price_series,
            "Volume": 1_000_000,
        }
    )

    class DummyLoader:
        def fetch_many(self, requests):
            return {req.symbol: template.copy() for req in requests}

    engine.loader = DummyLoader()
    result = engine.run()

    report = build_html_report(
        result.output_dir,
        result.metrics,
        result.monthly_returns,
        result.output_dir / "plots" / "equity_vs_benchmark.png",
        result.output_dir / "plots" / "drawdown.png",
        result.output_dir / "plots" / "monthly_heatmap.png",
        result.strategy_attribution,
        exposure_plot=result.output_dir / "plots" / "exposure.png",
        rolling_sharpe_plot=result.output_dir / "plots" / "rolling_sharpe.png",
        strategy_contribution_plot=result.output_dir / "plots" / "strategy_contribution.png",
        spy_comparison_plot=result.output_dir / "plots" / "spy_comparison.png",
    )

    assert report.exists()
    positions_df = pd.DataFrame({sym: df["position"] for sym, df in result.symbol_frames.items()})
    gross_exposure = positions_df.abs().sum(axis=1).max() if not positions_df.empty else 0.0
    assert gross_exposure <= config.risk.max_gross_exposure + 1e-6
