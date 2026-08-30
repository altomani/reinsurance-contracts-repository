from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from conftest import make_decision
from reinsurance_classifier.aggregation import aggregate_decisions
from reinsurance_classifier.audit_repair import repair_direct_evidence_gate
from reinsurance_classifier.output import (
    AuditRecord,
    AuditWriter,
    RecordStatus,
    read_audit_records,
    read_latest_records,
    recorded_cost,
)
from reinsurance_classifier.provider import ProviderCallResult


def test_repair_is_append_only_and_does_not_double_count_cost(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    csv_path = tmp_path / "latest.csv"
    decision = make_decision()
    call = ProviderCallResult(
        model_id="test/model",
        required_provider="test",
        decision=decision,
        input_tokens=10,
        output_tokens=10,
        cost_usd=0.01,
        latency_seconds=0.1,
        evidence_lines_valid=False,
        evidence_quotes_valid=False,
    )
    original = AuditRecord(
        record_id="one",
        year=2024,
        download_filename="one.txt",
        status=RecordStatus.QUALIFIED,
        prompt_version="test",
        created_at=AuditRecord.now(),
        metadata={},
        model_calls=[call],
        aggregate=aggregate_decisions([decision]),
        budget_charged_usd=0.01,
    )
    asyncio.run(AuditWriter(audit_path).append(original))

    assert asyncio.run(repair_direct_evidence_gate(audit_path, csv_path)) == 1
    records = read_audit_records(audit_path)
    latest = read_latest_records(audit_path)["one"]

    assert len(records) == 2
    assert latest.is_correction is True
    assert latest.status == RecordStatus.MANUAL_REVIEW
    assert latest.aggregate is not None and latest.aggregate.qualifies is False
    assert recorded_cost(records) == pytest.approx(0.01)
