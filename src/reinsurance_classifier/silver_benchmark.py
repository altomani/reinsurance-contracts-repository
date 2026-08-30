"""Evaluate a routed pilot against autonomous silver labels and build its cost gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .aggregation import aggregate_decisions
from .output import RecordStatus, read_latest_records


CLASSIFIED = {
    RecordStatus.QUALIFIED,
    RecordStatus.REJECTED,
    RecordStatus.MANUAL_REVIEW,
}


def evaluate(
    *,
    silver_path: Path,
    pilot_path: Path,
    confirmation_path: Path | None,
    prepared_total: int,
    remaining_authorized_usd: float,
    total_audit_budget_usd: float,
    prompt_version: str,
    pilot_reused: bool = False,
    quality_split: str = "holdout",
    contingency: float = 0.20,
) -> dict:
    if prepared_total < 1 or remaining_authorized_usd < 0 or total_audit_budget_usd <= 0:
        raise ValueError("invalid corpus or budget value")
    if contingency < 0.20:
        raise ValueError("contingency must be at least 20%")
    silver = {
        row["download_filename"]: row
        for row in csv.DictReader(silver_path.open(newline="", encoding="utf-8-sig"))
        if row.get("status") != "manual_review"
    }
    pilot = read_latest_records(pilot_path)
    records = [record for record in pilot.values() if record.status in CLASSIFIED]
    confirmations = (
        {
            record.download_filename: record
            for record in read_latest_records(confirmation_path).values()
        }
        if confirmation_path
        else {}
    )
    evaluated = {}
    missing_confirmations = 0
    for record in records:
        calls = list(record.model_calls)
        cost = record.budget_charged_usd
        if record.aggregate and record.aggregate.qualifies and len(calls) == 1:
            confirmation = confirmations.get(record.download_filename)
            if confirmation is None or not confirmation.model_calls:
                missing_confirmations += 1
                continue
            call = confirmation.model_calls[0]
            calls.append(call)
            cost += call.cost_usd or 0.0
        aggregate = aggregate_decisions([call.decision for call in calls])
        evaluated[record.download_filename] = (record, aggregate, calls, cost)

    split_metrics = {}
    for split in ("development", "holdout", "all"):
        pairs = []
        for filename, label in silver.items():
            if split != "all" and label.get("split") != split:
                continue
            result = evaluated.get(filename)
            if result is None:
                continue
            pairs.append((label["qualifies"] == "true", result[1].qualifies))
        split_metrics[split] = _metrics(pairs)

    costs = [result[3] for result in evaluated.values() if result[3]]
    mean_cost = sum(costs) / len(costs) if costs else 0.0
    p90_cost = _percentile(costs, 0.90)
    p95_cost = _percentile(costs, 0.95)
    conservative_unit = max(mean_cost, p90_cost)
    processed = set(evaluated)
    already_processed = len(processed) if pilot_reused else 0
    remaining = max(0, prepared_total - already_processed)
    remaining_cost = remaining * conservative_unit * (1 + contingency)
    if quality_split not in split_metrics:
        raise ValueError("quality_split must be development, holdout, or all")
    quality_metrics = split_metrics[quality_split]
    quality_pass = (
        quality_metrics["precision"] >= 0.95
        and quality_metrics["recall"] >= 0.90
    )
    pilot_size_pass = 100 <= len(evaluated) <= 200
    prompt_frozen = {record.prompt_version for record in records} == {prompt_version}
    budget_pass = remaining_cost <= remaining_authorized_usd
    full_run_permitted = all(
        (quality_pass, pilot_size_pass, prompt_frozen, bool(costs), budget_pass)
    )
    calls = [call for _, _, record_calls, _ in evaluated.values() for call in record_calls]
    return {
        "full_run_permitted": full_run_permitted,
        "label_provenance": "autonomous_silver_v1",
        "prompt_version": prompt_version,
        "quality_gate": {
            "passed": quality_pass,
            "thresholds": {"precision": 0.95, "recall": 0.90},
            "evaluated_split": quality_split,
            "development": split_metrics["development"],
            "holdout": split_metrics["holdout"],
            "all_resolved_silver": split_metrics["all"],
        },
        "pilot_gate": {
            "passed": pilot_size_pass,
            "classified_documents": len(evaluated),
            "prompt_frozen": prompt_frozen,
            "prompt_versions_seen": sorted({record.prompt_version for record in records}),
            "manual_review_rate": (
                sum(result[1].status.value == "manual_review" for result in evaluated.values())
                / len(evaluated)
                if evaluated
                else 0.0
            ),
            "escalation_rate": (
                sum(len(result[2]) > 1 for result in evaluated.values()) / len(evaluated)
                if evaluated
                else 0.0
            ),
            "provider_error_records": sum(
                record.status == RecordStatus.ERROR for record in pilot.values()
            ) + missing_confirmations,
            "hard_evidence_validation_rate": (
                sum(
                    call.evidence_lines_valid or call.evidence_quotes_valid
                    for call in calls
                )
                / len(calls)
                if calls
                else 0.0
            ),
        },
        "observed_costs": {
            "pilot_spent_usd": sum(costs),
            "mean_record_cost_usd": mean_cost,
            "p90_record_cost_usd": p90_cost,
            "p95_record_cost_usd": p95_cost,
            "conservative_unit_cost_usd": conservative_unit,
        },
        "corpus": {
            "prepared_supported_documents": prepared_total,
            "already_processed_documents": already_processed,
            "remaining_documents": remaining,
        },
        "forecast": {
            "contingency": contingency,
            "remaining_cost_usd": remaining_cost,
            "remaining_authorized_usd": remaining_authorized_usd,
            "cli_budget_usd": total_audit_budget_usd,
            "fits_authorized_remainder": budget_pass,
        },
    }


def _metrics(pairs: list[tuple[bool, bool]]) -> dict:
    tp = sum(expected and predicted for expected, predicted in pairs)
    fp = sum(not expected and predicted for expected, predicted in pairs)
    tn = sum(not expected and not predicted for expected, predicted in pairs)
    fn = sum(expected and not predicted for expected, predicted in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(pairs),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": precision,
        "precision_95_ci": _wilson(tp, tp + fp),
        "recall": recall,
        "recall_95_ci": _wilson(tp, tp + fn),
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1)]


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if not total:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--prepared-total", type=int, required=True)
    parser.add_argument("--remaining-authorized-usd", type=float, required=True)
    parser.add_argument("--total-audit-budget-usd", type=float, required=True)
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--pilot-reused", action="store_true")
    parser.add_argument(
        "--quality-split", choices=("development", "holdout", "all"), default="holdout"
    )
    parser.add_argument("--contingency", type=float, default=0.20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quality-override-report",
        type=Path,
        help="Use the quality gate from a separately frozen evaluation set.",
    )
    parser.add_argument(
        "--cost-stress-report",
        type=Path,
        help="Use a higher observed unit cost from a separate stress sample.",
    )
    args = parser.parse_args(argv)
    try:
        report = evaluate(
            silver_path=args.silver,
            pilot_path=args.pilot,
            confirmation_path=args.confirmation,
            prepared_total=args.prepared_total,
            remaining_authorized_usd=args.remaining_authorized_usd,
            total_audit_budget_usd=args.total_audit_budget_usd,
            prompt_version=args.prompt_version,
            pilot_reused=args.pilot_reused,
            quality_split=args.quality_split,
            contingency=args.contingency,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.quality_override_report:
        try:
            override = json.loads(
                args.quality_override_report.read_text(encoding="utf-8")
            )
            quality_gate = override["quality_gate"]
            if not isinstance(quality_gate, dict) or "passed" not in quality_gate:
                raise ValueError("override report has no valid quality_gate")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        quality_gate = dict(quality_gate)
        quality_gate["source_report"] = str(args.quality_override_report)
        report["quality_gate"] = quality_gate
    if args.cost_stress_report:
        try:
            stress = json.loads(args.cost_stress_report.read_text(encoding="utf-8"))
            stress_unit = float(stress["observed_costs"]["conservative_unit_cost_usd"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        if stress_unit > report["observed_costs"]["conservative_unit_cost_usd"]:
            report["observed_costs"]["conservative_unit_cost_usd"] = stress_unit
            report["observed_costs"]["cost_stress_source"] = str(args.cost_stress_report)
            report["forecast"]["remaining_cost_usd"] = (
                report["corpus"]["remaining_documents"]
                * stress_unit
                * (1 + report["forecast"]["contingency"])
            )
            report["forecast"]["fits_authorized_remainder"] = (
                report["forecast"]["remaining_cost_usd"]
                <= report["forecast"]["remaining_authorized_usd"]
            )
    report["full_run_permitted"] = all(
        (
            report["quality_gate"].get("passed") is True,
            report["pilot_gate"]["passed"] is True,
            report["pilot_gate"]["prompt_frozen"] is True,
            report["observed_costs"]["pilot_spent_usd"] > 0,
            report["forecast"]["fits_authorized_remainder"] is True,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
