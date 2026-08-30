from __future__ import annotations

import csv
from pathlib import Path

from conftest import make_decision
from reinsurance_classifier.aggregation import aggregate_decisions
from reinsurance_classifier.benchmark import evaluate
from reinsurance_classifier.models import BusinessBasis
from reinsurance_classifier.output import (
    AuditRecord,
    EvidencePackAudit,
    RecordStatus,
)
from reinsurance_classifier.provider import ProviderCallResult


def test_benchmark_reports_aggregate_model_criteria_and_operations(tmp_path: Path) -> None:
    positive = make_decision()
    negative = make_decision(business_basis=BusinessBasis.LIFE_LIKE)
    audit = tmp_path / "audit.jsonl"
    records = []
    for filename, decision, status in (
        ("positive.txt", positive, RecordStatus.QUALIFIED),
        ("negative.txt", negative, RecordStatus.REJECTED),
    ):
        call = ProviderCallResult(
            model_id="test/model",
            required_provider="test",
            downstream_provider="test",
            decision=decision,
            input_tokens=100,
            output_tokens=20,
            cost_usd=0.01,
            latency_seconds=0.5,
        )
        records.append(
            AuditRecord(
                record_id=f"2024:2:{filename}",
                year=2024,
                download_filename=filename,
                status=status,
                prompt_version="test-v1",
                created_at=AuditRecord.now(),
                metadata={},
                evidence_pack=EvidencePackAudit(
                    truncated=False,
                    normalized_chars=1_000,
                    selected_line_count=1,
                    selected_ranges=[(1, 1)],
                    categories_found=[],
                    request_chars=100,
                    estimated_input_tokens=34,
                ),
                model_calls=[call],
                aggregate=aggregate_decisions([decision]),
                budget_charged_usd=0.01,
            )
        )
    audit.write_text(
        "".join(record.model_dump_json() + "\n" for record in records),
        encoding="utf-8",
    )

    gold = tmp_path / "gold.csv"
    common = {
        "split": "holdout",
        "reviewer_1": "reviewer-a",
        "reviewer_2": "reviewer-b",
        "adjudicator": "adjudicator-c",
        "document_kind": "complete_contract",
        "is_reinsurance_contract": "yes",
        "relationship_term": "present",
        "business_covered_term": "present",
        "term_period_term": "present",
        "risk_economics_term": "present",
        "premium_term": "redacted",
        "overall_completeness": "sufficient",
        "placement_basis": "treaty",
        "government_basis": "private_market",
        "pool_exact_name": "",
        "pool_involvement": "none",
        "evidence_reinsurance": "L1: evidence",
        "evidence_completeness": "L1: evidence",
        "evidence_business": "L1: evidence",
        "evidence_placement": "L1: evidence",
        "evidence_government": "L1: evidence",
    }
    rows = [
        {
            **common,
            "download_filename": "positive.txt",
            "business_basis": "non_life",
            "primary_rejection_reason": "none",
            "qualifies": "true",
        },
        {
            **common,
            "download_filename": "negative.txt",
            "business_basis": "life_like",
            "primary_rejection_reason": "life_like_business",
            "qualifies": "false",
        },
    ]
    with gold.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report = evaluate(gold, audit, split="holdout")

    assert report["matched_rows"] == 2
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["criterion_accuracy"]["non_life_business"] == 1.0
    assert report["rejection_reason_accuracy"] == 1.0
    assert report["models"]["test/model"]["operations"]["calls"] == 2
    assert report["models"]["test/model"]["operations"]["cost_usd"] == 0.02
    assert report["evidence_line_coverage_by_length"]["lt_50k_chars"] == 1.0


def test_benchmark_refuses_single_review_seed(tmp_path: Path) -> None:
    gold = tmp_path / "gold.csv"
    row = {
        "download_filename": "sample.txt",
        "split": "development",
        "reviewer_1": "only-reviewer",
        "reviewer_2": "",
        "adjudicator": "",
        "qualifies": "false",
    }
    with gold.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    audit = tmp_path / "audit.jsonl"
    audit.write_text("", encoding="utf-8")

    import pytest

    with pytest.raises(ValueError, match="not adjudicated"):
        evaluate(gold, audit)
