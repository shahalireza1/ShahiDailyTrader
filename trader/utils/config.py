from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
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
    max_drawdown_stop: Optional[float] = 0.15
    drawdown_safe_fraction: float = 0.3
    max_gross_exposure: float = 0.6
    max_position_per_symbol: float = 0.25
    vol_target_annual: Optional[float] = None
    target_volatility: Optional[float] = None
    vol_lookback: int = 20
    position_mode: str = "fixed_fraction"
    position_fraction: float = 1.0
    kelly_safety: float = 0.5
    trade_cooldown_days: int = 1


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
    slippage_bps: float = 2.0
    starting_cash: float = 100_000.0
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    strategies: List[StrategyConfig] = field(default_factory=list)
    ensemble: Dict[str, Any] = field(default_factory=dict)
    risk: RiskConfig = field(default_factory=RiskConfig)
    output_dir: Path = Path("outputs")
    walkforward: WalkForwardConfig = field(default_factory=WalkForwardConfig)


def validate_config(cfg: Config) -> None:
    if not cfg.symbols:
        raise ValueError("At least one trading symbol must be provided.")
    if cfg.starting_cash <= 0:
        raise ValueError("Starting cash must be positive.")
    if cfg.fees_bps < 0 or cfg.slippage_bps < 0:
        raise ValueError("Fees and slippage basis points must be non-negative.")
    if cfg.risk.position_fraction <= 0 or cfg.risk.position_fraction > 1:
        raise ValueError("position_fraction must be in the interval (0, 1].")
    if cfg.risk.max_gross_exposure <= 0:
        raise ValueError("max_gross_exposure must be positive.")
    if cfg.risk.max_position_per_symbol <= 0:
        raise ValueError("max_position_per_symbol must be positive.")
    allowed_modes = {"backtest", "walkforward", "paper", "report"}
    if cfg.mode not in allowed_modes:
        raise ValueError(f"mode must be one of {sorted(allowed_modes)}")


def _merge_dict(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = default.copy()
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "dict") and callable(getattr(value, "dict")):
        try:
            return dict(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def load_config(path: Path) -> Config:
    if yaml is None:
        raise ImportError("PyYAML is required to load configuration files. Please install PyYAML.")

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open() as f:
        data = yaml.safe_load(f) or {}

    cfg = Config()
    raw = _merge_dict(cfg.__dict__, data)

    strategy_data = _to_dict(raw.get("strategy", {}) or {})
    strategies_data_raw = raw.get("strategies") or []
    risk_data = _to_dict(raw.get("risk", {}) or {})
    walk_data = _to_dict(raw.get("walkforward", {}) or {})
    ensemble_data = _to_dict(raw.get("ensemble", {}) or {})

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
    cfg.strategies = []
    for item in strategies_data_raw:
        mapping = _to_dict(item)
        if not mapping:
            continue
        cfg.strategies.append(
            StrategyConfig(name=str(mapping.get("name", cfg.strategy.name)), parameters=dict(mapping.get("parameters", {})))
        )
    if not cfg.strategies:
        cfg.strategies = [cfg.strategy]

    cfg.risk = RiskConfig(
        max_drawdown=float(risk_data.get("max_drawdown", cfg.risk.max_drawdown)),
        max_drawdown_stop=risk_data.get("max_drawdown_stop", cfg.risk.max_drawdown_stop),
        drawdown_safe_fraction=float(risk_data.get("drawdown_safe_fraction", cfg.risk.drawdown_safe_fraction)),
        max_gross_exposure=float(risk_data.get("max_gross_exposure", cfg.risk.max_gross_exposure)),
        max_position_per_symbol=float(risk_data.get("max_position_per_symbol", cfg.risk.max_position_per_symbol)),
        vol_target_annual=risk_data.get("vol_target_annual", cfg.risk.vol_target_annual),
        target_volatility=risk_data.get("target_volatility", risk_data.get("vol_target_annual", cfg.risk.target_volatility)),
        vol_lookback=int(risk_data.get("vol_lookback", cfg.risk.vol_lookback)),
        position_mode=str(risk_data.get("position_mode", cfg.risk.position_mode)),
        position_fraction=float(risk_data.get("position_fraction", cfg.risk.position_fraction)),
        kelly_safety=float(risk_data.get("kelly_safety", cfg.risk.kelly_safety)),
        trade_cooldown_days=int(risk_data.get("trade_cooldown_days", cfg.risk.trade_cooldown_days)),
    )

    cfg.ensemble = dict(ensemble_data)

    cfg.walkforward = WalkForwardConfig(
        train_window=int(walk_data.get("train_window", cfg.walkforward.train_window)),
        test_window=int(walk_data.get("test_window", cfg.walkforward.test_window)),
        step=int(walk_data.get("step", cfg.walkforward.step)),
    )

    validate_config(cfg)
    return cfg


def config_to_dict(config: Config) -> Dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if is_dataclass(value):
            return {k: convert(v) for k, v in asdict(value).items()}
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return convert(config)
