from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import make_decision
from reinsurance_classifier.aggregation import aggregate_decisions
from reinsurance_classifier.cli import _validate_full_run_gate
from reinsurance_classifier.forecast import build_forecast
from reinsurance_classifier.output import AuditRecord, RecordStatus
from reinsurance_classifier.provider import ProviderCallResult


def _write_jsonl(path: Path, records: list[AuditRecord]) -> None:
    path.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )


def test_forecast_requires_quality_review_cost_and_budget_gates(tmp_path: Path) -> None:
    decision = make_decision()
    preparation = []
    pilot = []
    for number in range(120):
        filename = f"sample-{number}.txt"
        preparation.append(
            AuditRecord(
                record_id=f"prep:{number}",
                year=2024,
                download_filename=filename,
                status=RecordStatus.PREPARED,
                prompt_version="classifier-v1",
                created_at=AuditRecord.now(),
                metadata={},
            )
        )
        if number < 100:
            call = ProviderCallResult(
                model_id="test/model",
                required_provider="test",
                decision=decision,
                input_tokens=100,
                output_tokens=20,
                cost_usd=0.01,
                latency_seconds=0.1,
            )
            pilot.append(
                AuditRecord(
                    record_id=f"pilot:{number}",
                    year=2024,
                    download_filename=filename,
                    status=RecordStatus.QUALIFIED,
                    prompt_version="classifier-v1",
                    created_at=AuditRecord.now(),
                    metadata={},
                    model_calls=[call],
                    aggregate=aggregate_decisions([decision]),
                    budget_charged_usd=0.01,
                )
            )
    prep_path = tmp_path / "prep.jsonl"
    pilot_path = tmp_path / "pilot.jsonl"
    benchmark_path = tmp_path / "benchmark.json"
    _write_jsonl(prep_path, preparation)
    _write_jsonl(pilot_path, pilot)
    benchmark_path.write_text(
        json.dumps(
            {
                "precision": 0.97,
                "recall": 0.92,
                "matched_rows": 60,
                "precision_95_ci": [0.9, 1.0],
                "recall_95_ci": [0.85, 0.98],
            }
        ),
        encoding="utf-8",
    )

    report = build_forecast(
        preparation_audit=prep_path,
        pilot_audit=pilot_path,
        benchmark_report=benchmark_path,
        available_credits_usd=1.0,
        cli_budget_usd=1.0,
        prompt_version="classifier-v1",
        positives_reviewed=True,
        sampled_negatives_reviewed=True,
    )

    assert report["full_run_permitted"] is True
    assert report["corpus"]["remaining_documents"] == 20
    assert report["forecast"]["remaining_cost_usd"] == pytest.approx(0.24)

    blocked = build_forecast(
        preparation_audit=prep_path,
        pilot_audit=pilot_path,
        benchmark_report=benchmark_path,
        available_credits_usd=0.10,
        cli_budget_usd=1.0,
        prompt_version="classifier-v1",
        positives_reviewed=False,
        sampled_negatives_reviewed=True,
    )
    assert blocked["full_run_permitted"] is False


def test_full_run_gate_must_match_prompt_and_budget(tmp_path: Path) -> None:
    path = tmp_path / "gate.json"
    path.write_text(
        json.dumps(
            {
                "full_run_permitted": True,
                "prompt_version": "classifier-v1",
                "forecast": {"cli_budget_usd": 5.0},
            }
        ),
        encoding="utf-8",
    )

    _validate_full_run_gate(path, prompt_version="classifier-v1", cli_budget_usd=4.0)
    with pytest.raises(ValueError, match="different prompt"):
        _validate_full_run_gate(path, prompt_version="classifier-v2", cli_budget_usd=4.0)
    with pytest.raises(ValueError, match="exceeds"):
        _validate_full_run_gate(path, prompt_version="classifier-v1", cli_budget_usd=6.0)
