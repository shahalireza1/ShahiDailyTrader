import types

from trader.utils import config as config_module


def test_load_config_handles_dataclass_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("symbols: []\n")

    cfg = config_module.load_config(config_path)

    assert cfg.strategy.name == "sma_rsi"
    assert cfg.strategies[0].name == cfg.strategy.name


def test_load_config_accepts_strategy_objects(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("")

    def fake_safe_load(_file):
        return {
            "symbols": ["AAPL"],
            "strategy": config_module.StrategyConfig(name="ensemble", parameters={"foo": 1}),
            "strategies": [config_module.StrategyConfig(name="momentum", parameters={"lookback": 5})],
            "risk": config_module.RiskConfig(max_drawdown=0.1),
            "walkforward": config_module.WalkForwardConfig(train_window=10, test_window=5, step=5),
            "ensemble": types.SimpleNamespace(weighting="equal"),
            "output_dir": str(tmp_path / "outputs"),
        }

    monkeypatch.setattr(config_module.yaml, "safe_load", fake_safe_load)

    cfg = config_module.load_config(config_path)

    assert cfg.strategy.name == "ensemble"
    assert cfg.strategy.parameters == {"foo": 1}
    assert cfg.strategies[0].name == "momentum"
    assert cfg.risk.max_drawdown == 0.1
    assert cfg.walkforward.train_window == 10
    assert cfg.ensemble.get("weighting") == "equal"
    assert cfg.output_dir == tmp_path / "outputs"
