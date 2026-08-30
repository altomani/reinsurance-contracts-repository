"""Append conservative corrections for legacy qualifications with invalid evidence."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .output import (
    AuditWriter,
    RecordStatus,
    read_latest_records,
    write_latest_csv,
)
from .runner import enforce_direct_evidence_gate


async def repair_direct_evidence_gate(jsonl_path: Path, csv_path: Path) -> int:
    """Downgrade unsupported qualifications without altering audit history."""

    writer = AuditWriter(jsonl_path)
    corrected = 0
    for record in read_latest_records(jsonl_path).values():
        if record.aggregate is None or not record.aggregate.qualifies:
            continue
        aggregate = enforce_direct_evidence_gate(record.aggregate, record.model_calls)
        if aggregate.qualifies:
            continue
        await writer.append(
            record.model_copy(
                update={
                    "status": RecordStatus.MANUAL_REVIEW,
                    "aggregate": aggregate,
                    "created_at": record.now(),
                    "budget_charged_usd": 0.0,
                    "is_correction": True,
                    "error": None,
                }
            )
        )
        corrected += 1
    write_latest_csv(jsonl_path, csv_path)
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    args = parser.parse_args()
    corrected = asyncio.run(repair_direct_evidence_gate(args.audit, args.csv))
    print(f"corrected={corrected}")


if __name__ == "__main__":
    main()
