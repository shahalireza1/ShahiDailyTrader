from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Type

import pandas as pd


class Strategy(ABC):
    name: str = "base"
    description: str = ""

    def __init__(self, **parameters) -> None:
        self.parameters = parameters

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame including a ``signal`` column aligned with price data."""

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        params = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        return f"{self.__class__.__name__}({params})"


class StrategyRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Type[Strategy]] = {}

    def register(self, strategy_cls: Type[Strategy]) -> None:
        if not issubclass(strategy_cls, Strategy):
            raise TypeError("Only Strategy subclasses can be registered")
        self._registry[strategy_cls.name] = strategy_cls

    def list_strategies(self) -> List[str]:
        return sorted(self._registry.keys())

    def create(self, name: str, **kwargs) -> Strategy:
        if name not in self._registry:
            available = ", ".join(self.list_strategies())
            raise KeyError(f"Unknown strategy '{name}'. Available: {available}")
        return self._registry[name](**kwargs)


registry = StrategyRegistry()


def register_strategy(cls: Type[Strategy]) -> Type[Strategy]:
    registry.register(cls)
    return cls


__all__ = ["Strategy", "StrategyRegistry", "registry", "register_strategy"]
