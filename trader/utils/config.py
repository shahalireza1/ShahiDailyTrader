from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - dependency is optional at import time for tests
    yaml = None


@dataclass
class StrategyConfig:
    name: str = "sma_rsi"
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskConfig:
    max_drawdown: float = 0.2
    target_volatility: Optional[float] = None
    vol_lookback: int = 20
    position_mode: str = "fixed_fraction"
    position_fraction: float = 1.0
    kelly_safety: float = 0.5


@dataclass
class WalkForwardConfig:
    train_window: int = 252
    test_window: int = 90
    step: int = 90


@dataclass
class Config:
    symbols: List[str] = field(default_factory=list)
    start: str = "2020-01-01"
    end: str = "2024-01-01"
    mode: str = "backtest"
    fees_bps: float = 1.0
    slippage_bps: float = 1.0
    starting_cash: float = 100_000.0
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    output_dir: Path = Path("outputs")
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)


def _merge_dict(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = default.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> Config:
    if yaml is None:
        raise ImportError("PyYAML is required to load configuration files. Please install PyYAML.")

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        data = yaml.safe_load(f) or {}

    cfg = Config()
    raw = _merge_dict(cfg.__dict__, data)

    strategy_data = raw.get("strategy", {}) or {}
    risk_data = raw.get("risk", {}) or {}
    walk_data = raw.get("walkforward", {}) or {}

    cfg.symbols = list(raw.get("symbols", []))
    cfg.start = raw.get("start", cfg.start)
    cfg.end = raw.get("end", cfg.end)
    cfg.mode = raw.get("mode", cfg.mode)
    cfg.fees_bps = float(raw.get("fees_bps", cfg.fees_bps))
    cfg.slippage_bps = float(raw.get("slippage_bps", cfg.slippage_bps))
    cfg.starting_cash = float(raw.get("starting_cash", cfg.starting_cash))
    cfg.output_dir = Path(raw.get("output_dir", cfg.output_dir))

    cfg.strategy = StrategyConfig(
        name=str(strategy_data.get("name", cfg.strategy.name)),
        parameters=dict(strategy_data.get("parameters", cfg.strategy.parameters)),
    )

    cfg.risk = RiskConfig(
        max_drawdown=float(risk_data.get("max_drawdown", cfg.risk.max_drawdown)),
        target_volatility=risk_data.get("target_volatility", cfg.risk.target_volatility),
        vol_lookback=int(risk_data.get("vol_lookback", cfg.risk.vol_lookback)),
        position_mode=str(risk_data.get("position_mode", cfg.risk.position_mode)),
        position_fraction=float(risk_data.get("position_fraction", cfg.risk.position_fraction)),
        kelly_safety=float(risk_data.get("kelly_safety", cfg.risk.kelly_safety)),
    )

    cfg.walkforward = WalkForwardConfig(
        train_window=int(walk_data.get("train_window", cfg.walkforward.train_window)),
        test_window=int(walk_data.get("test_window", cfg.walkforward.test_window)),
        step=int(walk_data.get("step", cfg.walkforward.step)),
    )

    return cfg
