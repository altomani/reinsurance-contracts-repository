"""Evaluate latest audit outputs against an adjudicated gold CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .aggregation import aggregate_decisions
from .output import read_latest_records


@dataclass(frozen=True)
class BinaryMetrics:
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


@dataclass
class _EvaluationBucket:
    pairs: list[tuple[bool, bool]] = field(default_factory=list)
    reviews: int = 0
    rejection_correct: int = 0
    rejection_total: int = 0
    criterion_correct: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    criterion_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    pool_name_correct: int = 0
    pool_name_total: int = 0
    pool_involvement_correct: int = 0
    pool_involvement_total: int = 0
    latencies: list[float] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 0
    retried_calls: int = 0


def _bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid gold boolean: {value!r}")


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def evaluate(gold_path: Path, audit_path: Path, *, split: str | None = None) -> dict:
    latest = read_latest_records(audit_path)
    gold_rows: list[dict[str, str]] = []
    with gold_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if not row.get("download_filename") or not row.get("qualifies"):
                continue
            if split and row.get("split") != split:
                continue
            _validate_gold_row(row)
            gold_rows.append(row)

    latest_by_filename = {
        record.download_filename: record for record in latest.values()
    }
    aggregate_bucket = _EvaluationBucket()
    model_buckets: dict[str, _EvaluationBucket] = defaultdict(_EvaluationBucket)
    criterion_fields = {
        "document_contract": ("document_kind", "is_reinsurance_contract"),
        "main_terms": (
            "relationship_term",
            "business_covered_term",
            "term_period_term",
            "risk_economics_term",
            "premium_term",
            "overall_completeness",
        ),
        "non_life_business": ("business_basis",),
        "treaty_placement": ("placement_basis",),
        "private_market": ("government_basis",),
    }
    evidence_coverage: dict[str, list[bool]] = defaultdict(list)
    failed_records = 0
    for gold in gold_rows:
        filename = gold["download_filename"]
        record = latest_by_filename.get(filename)
        if record is None:
            continue
        if record.aggregate is None:
            if record.status.value in {"error", "budget_exhausted"}:
                failed_records += 1
            continue
        predicted = record.aggregate.qualifies
        expected = _bool(gold["qualifies"])
        aggregate_bucket.pairs.append((expected, predicted))
        if record.status.value == "manual_review":
            aggregate_bucket.reviews += 1
        _score_consensus(aggregate_bucket, record.aggregate, gold, criterion_fields)
        _score_evidence_coverage(evidence_coverage, record, gold)

        for call in record.model_calls:
            bucket = model_buckets[call.model_id]
            single = aggregate_decisions([call.decision])
            bucket.pairs.append((expected, single.qualifies))
            if single.status.value == "manual_review":
                bucket.reviews += 1
            _score_consensus(bucket, single, gold, criterion_fields)
            expected_name = gold.get("pool_exact_name", "").strip()
            if expected_name:
                bucket.pool_name_total += 1
                if call.decision.pool_or_scheme.exact_name == expected_name:
                    bucket.pool_name_correct += 1
            expected_involvement = gold.get("pool_involvement", "").strip()
            if expected_involvement:
                bucket.pool_involvement_total += 1
                if call.decision.pool_or_scheme.involvement.value == expected_involvement:
                    bucket.pool_involvement_correct += 1
            bucket.latencies.append(call.latency_seconds)
            bucket.input_tokens += call.input_tokens
            bucket.output_tokens += call.output_tokens
            bucket.cost_usd += call.cost_usd or 0.0
            bucket.calls += 1
            bucket.retried_calls += int(call.retry_count > 0)

    aggregate_summary = _bucket_summary(aggregate_bucket)
    report = {
        "gold_rows": len(gold_rows),
        "matched_rows": len(aggregate_bucket.pairs),
        **aggregate_summary,
        "models": {
            model_id: _bucket_summary(bucket)
            for model_id, bucket in sorted(model_buckets.items())
        },
        "failed_records": failed_records,
        "evidence_line_coverage_by_length": {
            length_bin: sum(values) / len(values) if values else None
            for length_bin, values in sorted(evidence_coverage.items())
        },
    }
    return report


def _validate_gold_row(row: dict[str, str]) -> None:
    required = (
        "reviewer_1",
        "reviewer_2",
        "adjudicator",
        "document_kind",
        "is_reinsurance_contract",
        "relationship_term",
        "business_covered_term",
        "term_period_term",
        "risk_economics_term",
        "premium_term",
        "overall_completeness",
        "business_basis",
        "placement_basis",
        "government_basis",
        "primary_rejection_reason",
        "evidence_reinsurance",
        "evidence_completeness",
        "evidence_business",
        "evidence_placement",
        "evidence_government",
    )
    missing = [field for field in required if not row.get(field, "").strip()]
    if missing:
        raise ValueError(
            f"gold row {row.get('download_filename')!r} is not adjudicated; "
            f"missing: {', '.join(missing)}"
        )
    reviewers = {
        row["reviewer_1"].strip(),
        row["reviewer_2"].strip(),
        row["adjudicator"].strip(),
    }
    if len(reviewers) != 3:
        raise ValueError("gold rows require two distinct reviewers and a distinct adjudicator")
    criterion_fields = {
        "document_contract": ("document_kind", "is_reinsurance_contract"),
        "main_terms": (
            "relationship_term",
            "business_covered_term",
            "term_period_term",
            "risk_economics_term",
            "premium_term",
            "overall_completeness",
        ),
        "non_life_business": ("business_basis",),
        "treaty_placement": ("placement_basis",),
        "private_market": ("government_basis",),
    }
    expected_values = [
        _expected_criterion(row, output_field, sources)
        for output_field, sources in criterion_fields.items()
    ]
    if any(value is None for value in expected_values):
        raise ValueError(f"gold row {row['download_filename']!r} has invalid criterion values")
    derived_qualifies = all(value == "pass" for value in expected_values)
    if _bool(row["qualifies"]) != derived_qualifies:
        raise ValueError(
            f"gold row {row['download_filename']!r} has a qualifies value "
            "inconsistent with the five-gate conjunction"
        )


def _score_consensus(
    bucket: _EvaluationBucket,
    aggregate,
    gold: dict[str, str],
    criterion_fields: dict[str, tuple[str, ...]],
) -> None:
    consensus = aggregate.consensus
    for output_field, gold_sources in criterion_fields.items():
        expected_criterion = _expected_criterion(gold, output_field, gold_sources)
        if expected_criterion is None:
            continue
        bucket.criterion_total[output_field] += 1
        if getattr(consensus, output_field).value == expected_criterion:
            bucket.criterion_correct[output_field] += 1
    expected_reason = gold.get("primary_rejection_reason", "").strip()
    if expected_reason:
        bucket.rejection_total += 1
        if aggregate.primary_rejection_reason.value == expected_reason:
            bucket.rejection_correct += 1


def _bucket_summary(bucket: _EvaluationBucket) -> dict:
    tp = sum(expected and predicted for expected, predicted in bucket.pairs)
    fp = sum(not expected and predicted for expected, predicted in bucket.pairs)
    tn = sum(not expected and not predicted for expected, predicted in bucket.pairs)
    fn = sum(expected and not predicted for expected, predicted in bucket.pairs)
    metrics = BinaryMetrics(tp, fp, tn, fn)
    return {
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": metrics.precision,
        "precision_95_ci": _wilson(tp, tp + fp),
        "recall": metrics.recall,
        "recall_95_ci": _wilson(tp, tp + fn),
        "f1": metrics.f1,
        "review_rate": bucket.reviews / len(bucket.pairs) if bucket.pairs else 0.0,
        "criterion_accuracy": {
            field: (
                bucket.criterion_correct[field] / bucket.criterion_total[field]
                if bucket.criterion_total[field]
                else None
            )
            for field in (
                "document_contract",
                "main_terms",
                "non_life_business",
                "treaty_placement",
                "private_market",
            )
        },
        "rejection_reason_accuracy": (
            bucket.rejection_correct / bucket.rejection_total
            if bucket.rejection_total
            else None
        ),
        "pool_exact_name_accuracy": (
            bucket.pool_name_correct / bucket.pool_name_total
            if bucket.pool_name_total
            else None
        ),
        "pool_involvement_accuracy": (
            bucket.pool_involvement_correct / bucket.pool_involvement_total
            if bucket.pool_involvement_total
            else None
        ),
        "operations": {
            "calls": bucket.calls,
            "retry_rate": bucket.retried_calls / bucket.calls if bucket.calls else 0.0,
            "mean_latency_seconds": (
                sum(bucket.latencies) / len(bucket.latencies) if bucket.latencies else None
            ),
            "p95_latency_seconds": _percentile(bucket.latencies, 0.95),
            "input_tokens": bucket.input_tokens,
            "output_tokens": bucket.output_tokens,
            "cost_usd": bucket.cost_usd,
        },
    }


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


_LINE_REFERENCE = re.compile(r"L(\d+)(?:-L?(\d+))?", re.I)


def _score_evidence_coverage(
    coverage: dict[str, list[bool]], record, gold: dict[str, str]
) -> None:
    if record.evidence_pack is None:
        return
    references: set[int] = set()
    for field in (
        "evidence_reinsurance",
        "evidence_completeness",
        "evidence_business",
        "evidence_placement",
        "evidence_government",
    ):
        for match in _LINE_REFERENCE.finditer(gold.get(field, "")):
            start = int(match.group(1))
            end = int(match.group(2) or start)
            references.update(range(start, end + 1))
    if not references:
        return
    selected = {
        number
        for start, end in record.evidence_pack.selected_ranges
        for number in range(start, end + 1)
    }
    chars = record.evidence_pack.normalized_chars
    if chars < 50_000:
        length_bin = "lt_50k_chars"
    elif chars < 250_000:
        length_bin = "50k_250k_chars"
    elif chars < 1_000_000:
        length_bin = "250k_1m_chars"
    else:
        length_bin = "gte_1m_chars"
    coverage[length_bin].append(references.issubset(selected))


def _expected_criterion(
    gold: dict[str, str], output_field: str, sources: tuple[str, ...]
) -> str | None:
    if output_field == "document_contract":
        kind = gold.get("document_kind", "")
        reinsurance = gold.get("is_reinsurance_contract", "")
        passing_kinds = {"complete_contract", "nearly_complete_contract"}
        if kind in passing_kinds and reinsurance == "yes":
            return "pass"
        if (kind not in passing_kinds and kind != "unclear") or reinsurance == "no":
            return "fail"
        return "unclear"
    if output_field == "main_terms":
        values = [gold.get(source, "") for source in sources]
        if not all(values):
            return None
        acceptable = {"present", "redacted"}
        if gold.get("overall_completeness") == "sufficient" and all(
            value in acceptable for value in values[:-1]
        ):
            return "pass"
        if gold.get("overall_completeness") == "insufficient" or "missing" in values:
            return "fail"
        return "unclear"
    source = gold.get(sources[0], "")
    mapping = {
        "non_life_business": {
            "non_life": "pass",
            "life_like": "fail",
            "mixed": "fail",
            "unclear": "unclear",
        },
        "treaty_placement": {
            "treaty": "pass",
            "facultative": "fail",
            "mixed": "fail",
            "unclear": "unclear",
        },
        "private_market": {
            "private_market": "pass",
            "statutory_government_scheme": "fail",
            "unclear": "unclear",
        },
    }
    return mapping[output_field].get(source)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--split")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = evaluate(args.gold, args.audit, split=args.split)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
