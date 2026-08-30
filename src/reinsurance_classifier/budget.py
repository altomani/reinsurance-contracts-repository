"""Concurrency-safe hard dollar budget reservations."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetSnapshot:
    ceiling_usd: float
    spent_usd: float
    reserved_usd: float

    @property
    def available_usd(self) -> float:
        return max(0.0, self.ceiling_usd - self.spent_usd - self.reserved_usd)


class BudgetLedger:
    def __init__(self, ceiling_usd: float, *, initial_spent_usd: float = 0.0) -> None:
        if ceiling_usd <= 0:
            raise ValueError("budget ceiling must be positive")
        if initial_spent_usd < 0:
            raise ValueError("initial spent cost cannot be negative")
        self.ceiling_usd = float(ceiling_usd)
        self._spent_usd = float(initial_spent_usd)
        self._reservations: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def reserve(self, estimated_cost_usd: float) -> str:
        if estimated_cost_usd <= 0:
            raise ValueError("estimated request cost must be positive")
        async with self._lock:
            committed = self._spent_usd + sum(self._reservations.values())
            if committed + estimated_cost_usd > self.ceiling_usd + 1e-12:
                raise BudgetExceeded(
                    f"request reserve ${estimated_cost_usd:.6f} would exceed "
                    f"the ${self.ceiling_usd:.2f} hard budget"
                )
            reservation_id = uuid.uuid4().hex
            self._reservations[reservation_id] = float(estimated_cost_usd)
            return reservation_id

    async def settle(self, reservation_id: str, actual_cost_usd: float | None) -> float:
        async with self._lock:
            try:
                reserved = self._reservations.pop(reservation_id)
            except KeyError as exc:
                raise ValueError("unknown or already settled reservation") from exc
            # Unknown-cost failures and responses remain charged at the conservative
            # reserve. A reported cost replaces the reserve, even when higher.
            charged = reserved if actual_cost_usd is None else max(0.0, actual_cost_usd)
            self._spent_usd += charged
            return charged

    async def snapshot(self) -> BudgetSnapshot:
        async with self._lock:
            return BudgetSnapshot(
                ceiling_usd=self.ceiling_usd,
                spent_usd=self._spent_usd,
                reserved_usd=sum(self._reservations.values()),
            )
