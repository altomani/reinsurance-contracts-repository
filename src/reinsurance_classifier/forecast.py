"""Build the conservative post-pilot cost and full-run gate report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .output import RecordStatus, read_latest_records


CLASSIFIED_STATUSES = {
    RecordStatus.QUALIFIED,
    RecordStatus.REJECTED,
    RecordStatus.MANUAL_REVIEW,
}


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1)]


def build_forecast(
    *,
    preparation_audit: Path,
    pilot_audit: Path,
    benchmark_report: Path,
    available_credits_usd: float,
    cli_budget_usd: float,
    prompt_version: str,
    positives_reviewed: bool,
    sampled_negatives_reviewed: bool,
    contingency: float = 0.20,
) -> dict:
    if available_credits_usd < 0 or cli_budget_usd <= 0:
        raise ValueError("credits must be nonnegative and CLI budget must be positive")
    if contingency < 0.20:
        raise ValueError("contingency must be at least 20%")
    preparation = read_latest_records(preparation_audit)
    pilot = read_latest_records(pilot_audit)
    benchmark = json.loads(benchmark_report.read_text(encoding="utf-8"))

    prepared_supported = sum(
        record.status == RecordStatus.PREPARED for record in preparation.values()
    )
    pilot_records = [
        record for record in pilot.values() if record.status in CLASSIFIED_STATUSES
    ]
    processed_filenames = {record.download_filename for record in pilot_records}
    prepared_filenames = {
        record.download_filename
        for record in preparation.values()
        if record.status == RecordStatus.PREPARED
    }
    remaining = len(prepared_filenames - processed_filenames)
    record_costs = [
        record.budget_charged_usd
        if record.budget_charged_usd
        else sum(call.cost_usd or 0.0 for call in record.model_calls)
        for record in pilot_records
    ]
    record_costs = [cost for cost in record_costs if cost > 0]
    mean_cost = sum(record_costs) / len(record_costs) if record_costs else 0.0
    p90_cost = _percentile(record_costs, 0.90)
    p95_cost = _percentile(record_costs, 0.95)
    conservative_unit_cost = max(mean_cost, p90_cost)
    forecast_cost = remaining * conservative_unit_cost * (1 + contingency)

    calls = [call for record in pilot_records for call in record.model_calls]
    escalation_rate = (
        sum(len(record.model_calls) > 1 for record in pilot_records)
        / len(pilot_records)
        if pilot_records
        else 0.0
    )
    retry_rate = (
        sum(call.retry_count > 0 for call in calls) / len(calls) if calls else 0.0
    )
    prompt_versions = {record.prompt_version for record in pilot_records}
    prompt_frozen = prompt_versions == {prompt_version}
    quality_pass = (
        benchmark.get("precision", 0.0) >= 0.95
        and benchmark.get("recall", 0.0) >= 0.90
        and benchmark.get("matched_rows", 0) > 0
    )
    pilot_size_pass = 100 <= len(pilot_records) <= 200
    forecast_fits = forecast_cost <= min(available_credits_usd, cli_budget_usd)
    full_run_permitted = all(
        (
            quality_pass,
            pilot_size_pass,
            positives_reviewed,
            sampled_negatives_reviewed,
            prompt_frozen,
            bool(record_costs),
            forecast_fits,
        )
    )
    return {
        "full_run_permitted": full_run_permitted,
        "prompt_version": prompt_version,
        "quality_gate": {
            "passed": quality_pass,
            "precision": benchmark.get("precision"),
            "recall": benchmark.get("recall"),
            "precision_95_ci": benchmark.get("precision_95_ci"),
            "recall_95_ci": benchmark.get("recall_95_ci"),
            "matched_rows": benchmark.get("matched_rows", 0),
        },
        "pilot_gate": {
            "passed": pilot_size_pass,
            "classified_documents": len(pilot_records),
            "positives_reviewed": positives_reviewed,
            "sampled_negatives_reviewed": sampled_negatives_reviewed,
            "prompt_frozen": prompt_frozen,
            "prompt_versions_seen": sorted(prompt_versions),
        },
        "corpus": {
            "prepared_supported_documents": prepared_supported,
            "remaining_documents": remaining,
        },
        "observed_costs": {
            "records_with_cost": len(record_costs),
            "mean_record_cost_usd": mean_cost,
            "p90_record_cost_usd": p90_cost,
            "p95_record_cost_usd": p95_cost,
            "conservative_unit_cost_usd": conservative_unit_cost,
            "escalation_rate": escalation_rate,
            "retry_rate": retry_rate,
        },
        "forecast": {
            "contingency": contingency,
            "remaining_cost_usd": forecast_cost,
            "available_credits_usd": available_credits_usd,
            "cli_budget_usd": cli_budget_usd,
            "fits_both_limits": forecast_fits,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-audit", type=Path, required=True)
    parser.add_argument("--pilot-audit", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--available-credits-usd", type=float, required=True)
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--prompt-version", required=True)
    parser.add_argument("--positives-reviewed", action="store_true")
    parser.add_argument("--sampled-negatives-reviewed", action="store_true")
    parser.add_argument("--contingency", type=float, default=0.20)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/full-run-gate.json")
    )
    args = parser.parse_args(argv)
    try:
        report = build_forecast(
            preparation_audit=args.preparation_audit,
            pilot_audit=args.pilot_audit,
            benchmark_report=args.benchmark_report,
            available_credits_usd=args.available_credits_usd,
            cli_budget_usd=args.budget_usd,
            prompt_version=args.prompt_version,
            positives_reviewed=args.positives_reviewed,
            sampled_negatives_reviewed=args.sampled_negatives_reviewed,
            contingency=args.contingency,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
