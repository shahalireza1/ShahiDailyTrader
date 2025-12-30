from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Order:
    symbol: str
    quantity: float
    side: str  # BUY / SELL
    price: float | None = None


class ExecutionProvider(ABC):
    """Abstract execution/broker adapter."""

    @abstractmethod
    def submit(self, order: Order) -> Dict[str, Any]:
        raise NotImplementedError


class PaperBroker(ExecutionProvider):
    def submit(self, order: Order) -> Dict[str, Any]:  # pragma: no cover - placeholder
        return {"status": "filled", "order": order}


class LiveBrokerPlaceholder(ExecutionProvider):
    def submit(self, order: Order) -> Dict[str, Any]:  # pragma: no cover - placeholder
        return {"status": "queued", "message": "Integrate broker SDK here", "order": order}
