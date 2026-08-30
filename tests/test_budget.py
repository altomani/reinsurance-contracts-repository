from __future__ import annotations

import asyncio

import pytest

from reinsurance_classifier.budget import BudgetExceeded, BudgetLedger


def test_budget_reservations_stop_before_the_next_request() -> None:
    async def scenario() -> None:
        ledger = BudgetLedger(0.10)
        first = await ledger.reserve(0.06)
        with pytest.raises(BudgetExceeded):
            await ledger.reserve(0.05)
        await ledger.settle(first, 0.02)
        second = await ledger.reserve(0.05)
        await ledger.settle(second, None)
        snapshot = await ledger.snapshot()
        assert snapshot.spent_usd == pytest.approx(0.07)
        assert snapshot.reserved_usd == 0

    asyncio.run(scenario())


def test_concurrent_reservations_cannot_oversubscribe_budget() -> None:
    async def scenario() -> None:
        ledger = BudgetLedger(0.05)

        async def attempt() -> bool:
            try:
                await ledger.reserve(0.03)
            except BudgetExceeded:
                return False
            return True

        outcomes = await asyncio.gather(attempt(), attempt())
        assert sorted(outcomes) == [False, True]
        snapshot = await ledger.snapshot()
        assert snapshot.reserved_usd == pytest.approx(0.03)

    asyncio.run(scenario())
