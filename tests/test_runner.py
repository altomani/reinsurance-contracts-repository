from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from conftest import make_decision
from reinsurance_classifier.extraction import RequestLimits
from reinsurance_classifier.metadata import MetadataRecord, SourceStatus
from reinsurance_classifier.models import BusinessBasis, Certainty
from reinsurance_classifier.output import RecordStatus, read_audit_records, read_latest_records
from reinsurance_classifier.provider import ModelRoute, ProviderCallResult
from reinsurance_classifier.runner import RunnerConfig, run_records


ROUTES = (
    ModelRoute("one", "test/one", "provider-one"),
    ModelRoute("two", "test/two", "provider-two"),
    ModelRoute("three", "test/three", "provider-three"),
)


class FakeBackend:
    def __init__(self, decisions: list, *, costs: list[float] | None = None) -> None:
        self.decisions = list(decisions)
        self.costs = list(costs or [0.01] * len(decisions))
        self.calls: list[str] = []
        self.failures: list[Exception] = []
        self.success_count = 0

    async def classify(
        self, evidence_pack, route, *, prompt_text: str, max_output_tokens: int
    ) -> ProviderCallResult:
        self.calls.append(route.model_id)
        if self.failures:
            raise self.failures.pop(0)
        index = self.success_count
        self.success_count += 1
        return ProviderCallResult(
            model_id=route.model_id,
            required_provider=route.provider_slug,
            downstream_provider=route.provider_slug,
            decision=self.decisions[index],
            input_tokens=100,
            output_tokens=20,
            cost_usd=self.costs[index],
            latency_seconds=0.01,
        )


def _record(tmp_path: Path) -> MetadataRecord:
    source = tmp_path / "sample.txt"
    source.write_text(
        "REINSURANCE AGREEMENT\nPremium is redacted\nAutomatic treaty\nIn witness",
        encoding="utf-8",
    )
    return MetadataRecord(
        record_id="2024:2:sample.txt",
        year=2024,
        row_number=2,
        download_filename="sample.txt",
        source_path=source,
        source_status=SourceStatus.SUPPORTED,
        metadata={"downloadFilename": "sample.txt", "description": "sample"},
    )


def _config(tmp_path: Path, **changes) -> RunnerConfig:
    values = {
        "prompt_text": "classify",
        "prompt_version": "test-v1",
        "routes": ROUTES,
        "limits": RequestLimits(),
        "budget_usd": 1.0,
        "request_cost_reserve_usd": 0.05,
        "concurrency": 2,
        "max_retries": 0,
        "dry_run": False,
        "benchmark_all": False,
        "resume": True,
        "jsonl_path": tmp_path / "audit.jsonl",
        "csv_path": tmp_path / "latest.csv",
    }
    values.update(changes)
    return RunnerConfig(**values)


def test_routing_uses_second_for_low_certainty_and_third_for_disagreement(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        [
            make_decision(certainty=Certainty.LOW),
            make_decision(business_basis=BusinessBasis.LIFE_LIKE),
            make_decision(),
        ]
    )

    summary = asyncio.run(run_records([_record(tmp_path)], backend, _config(tmp_path)))
    record = next(iter(read_latest_records(tmp_path / "audit.jsonl").values()))

    assert backend.calls == [route.model_id for route in ROUTES]
    assert record.status == RecordStatus.QUALIFIED
    assert record.aggregate is not None and record.aggregate.qualifies is True
    assert record.budget_charged_usd == pytest.approx(0.03)
    assert summary.spent_usd == pytest.approx(0.03)
    assert (tmp_path / "latest.csv").exists()


def test_budget_exhaustion_preserves_partial_paid_calls(tmp_path: Path) -> None:
    backend = FakeBackend([make_decision(certainty=Certainty.LOW)])
    config = _config(
        tmp_path,
        budget_usd=0.025,
        request_cost_reserve_usd=0.02,
    )

    asyncio.run(run_records([_record(tmp_path)], backend, config))
    record = read_audit_records(tmp_path / "audit.jsonl")[-1]

    assert record.status == RecordStatus.BUDGET_EXHAUSTED
    assert len(record.model_calls) == 1
    assert record.budget_charged_usd == pytest.approx(0.01)


def test_transient_retry_cost_is_included_in_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend([make_decision(), make_decision()])
    backend.failures.append(TimeoutError("temporary timeout"))

    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("reinsurance_classifier.runner.asyncio.sleep", no_sleep)
    config = _config(tmp_path, max_retries=1)
    asyncio.run(run_records([_record(tmp_path)], backend, config))
    record = read_audit_records(tmp_path / "audit.jsonl")[-1]

    assert record.status == RecordStatus.QUALIFIED
    assert record.model_calls[0].retry_count == 1
    assert record.budget_charged_usd == pytest.approx(0.07)


def test_dry_run_never_calls_provider_and_terminal_resume_skips(tmp_path: Path) -> None:
    config = _config(tmp_path, dry_run=True)
    first = asyncio.run(run_records([_record(tmp_path)], None, config))
    assert first.counts == {"prepared": 1}

    # Prepared is intentionally not terminal, so a later real run processes it.
    backend = FakeBackend([make_decision(), make_decision()])
    real_config = _config(tmp_path)
    asyncio.run(run_records([_record(tmp_path)], backend, real_config))
    second = asyncio.run(run_records([_record(tmp_path)], backend, real_config))
    assert second.skipped_by_resume == 1
    assert len(backend.calls) == 2


def test_high_certainty_positive_is_confirmed_by_second_model(tmp_path: Path) -> None:
    backend = FakeBackend([make_decision(), make_decision()])

    asyncio.run(run_records([_record(tmp_path)], backend, _config(tmp_path)))
    record = next(iter(read_latest_records(tmp_path / "audit.jsonl").values()))

    assert backend.calls == [ROUTES[0].model_id, ROUTES[1].model_id]
    assert record.aggregate is not None and record.aggregate.qualifies is True


def test_positive_without_any_directly_valid_evidence_is_manual_review(
    tmp_path: Path,
) -> None:
    backend = FakeBackend([make_decision(), make_decision()])
    original_classify = backend.classify

    async def classify_with_invalid_evidence(*args, **kwargs) -> ProviderCallResult:
        result = await original_classify(*args, **kwargs)
        return result.model_copy(
            update={
                "evidence_lines_valid": False,
                "evidence_quotes_valid": False,
                "evidence_validation_errors": ["ungrounded evidence"],
            }
        )

    backend.classify = classify_with_invalid_evidence  # type: ignore[method-assign]
    asyncio.run(run_records([_record(tmp_path)], backend, _config(tmp_path)))
    record = next(iter(read_latest_records(tmp_path / "audit.jsonl").values()))

    assert record.status == RecordStatus.MANUAL_REVIEW
    assert record.aggregate is not None
    assert record.aggregate.qualifies is False
    assert record.aggregate.direct_evidence_valid is False
