from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import numpy as np
import pandas as pd


class PipelineDiagnostics:
    """Collects pipeline block information across signal -> weight stages."""

    def __init__(self, eps: float = 1e-9) -> None:
        self.eps = eps
        self.block_reason_counts: Dict[str, int] = defaultdict(int)

    def record_block(self, reason: str, mask: Iterable[bool] | pd.Series | pd.DataFrame) -> None:
        if mask is None:
            return
        if isinstance(mask, pd.DataFrame):
            count = int(mask.astype(bool).to_numpy().sum())
        else:
            arr = np.array(mask, dtype=bool)
            count = int(arr.sum())
        if count:
            self.block_reason_counts[reason] += count

    def record_transition(self, upstream: pd.Series, downstream: pd.Series, reason: str) -> None:
        mask = (upstream.abs() > self.eps) & (downstream.abs() <= self.eps)
        self.record_block(reason, mask)

    def export(self) -> Dict[str, int]:
        return dict(self.block_reason_counts)
