"""Append-only audit records, resume state, and filter-friendly CSV output."""

from __future__ import annotations

import asyncio
import csv
import json
import os
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Iterable

from pydantic import Field

from .aggregation import AggregatedClassification
from .models import StrictModel
from .provider import ProviderCallResult


class RecordStatus(StrEnum):
    PREPARED = "prepared"
    QUALIFIED = "qualified"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"
    MISSING_FILE = "missing_file"
    SKIPPED_PDF = "skipped_pdf"
    UNSUPPORTED_FILE = "unsupported_file"
    ERROR = "error"
    BUDGET_EXHAUSTED = "budget_exhausted"


TERMINAL_STATUSES = {
    RecordStatus.QUALIFIED,
    RecordStatus.REJECTED,
    RecordStatus.MANUAL_REVIEW,
    RecordStatus.MISSING_FILE,
    RecordStatus.SKIPPED_PDF,
    RecordStatus.UNSUPPORTED_FILE,
}


class EvidencePackAudit(StrictModel):
    truncated: bool
    normalized_chars: int = Field(ge=0)
    selected_line_count: int = Field(ge=0)
    selected_ranges: list[tuple[int, int]]
    categories_found: list[str]
    request_chars: int = Field(ge=0)
    estimated_input_tokens: int = Field(ge=0)


class AuditRecord(StrictModel):
    record_id: str
    year: int
    download_filename: str
    status: RecordStatus
    prompt_version: str
    created_at: str
    metadata: dict[str, str]
    source_sha256: str | None = None
    evidence_pack: EvidencePackAudit | None = None
    model_calls: list[ProviderCallResult] = Field(default_factory=list)
    aggregate: AggregatedClassification | None = None
    budget_charged_usd: float = Field(default=0.0, ge=0)
    is_correction: bool = False
    error: str | None = None

    @classmethod
    def now(cls) -> str:
        return datetime.now(UTC).isoformat()


def read_audit_records(path: Path) -> list[AuditRecord]:
    records: list[AuditRecord] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = AuditRecord.model_validate_json(line)
            except Exception as exc:
                raise ValueError(f"invalid audit JSONL at {path}:{line_number}: {exc}") from exc
            records.append(record)
    return records


def read_latest_records(path: Path) -> dict[str, AuditRecord]:
    latest: dict[str, AuditRecord] = {}
    for record in read_audit_records(path):
        latest[record.record_id] = record
    return latest


def recorded_cost(records: Iterable[AuditRecord]) -> float:
    total = 0.0
    for record in records:
        if record.is_correction:
            continue
        if record.budget_charged_usd:
            total += record.budget_charged_usd
        else:
            total += sum(call.cost_usd or 0.0 for call in record.model_calls)
    return total


class AuditWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()

    async def append(self, record: AuditRecord) -> None:
        payload = record.model_dump_json() + "\n"
        async with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())


CSV_FIELDS = (
    "record_id",
    "year",
    "download_filename",
    "status",
    "qualifies",
    "direct_evidence_valid",
    "primary_rejection_reason",
    "document_contract",
    "main_terms",
    "non_life_business",
    "treaty_placement",
    "private_market",
    "model_count",
    "models",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "budget_charged_usd",
    "prompt_version",
    "error",
)


def write_latest_csv(jsonl_path: Path, csv_path: Path) -> None:
    latest = read_latest_records(jsonl_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in latest.values():
            aggregate = record.aggregate
            consensus = aggregate.consensus if aggregate else None
            writer.writerow(
                {
                    "record_id": record.record_id,
                    "year": record.year,
                    "download_filename": record.download_filename,
                    "status": record.status,
                    "qualifies": aggregate.qualifies if aggregate else "",
                    "direct_evidence_valid": (
                        aggregate.direct_evidence_valid if aggregate else ""
                    ),
                    "primary_rejection_reason": (
                        aggregate.primary_rejection_reason if aggregate else ""
                    ),
                    "document_contract": consensus.document_contract if consensus else "",
                    "main_terms": consensus.main_terms if consensus else "",
                    "non_life_business": (
                        consensus.non_life_business if consensus else ""
                    ),
                    "treaty_placement": consensus.treaty_placement if consensus else "",
                    "private_market": consensus.private_market if consensus else "",
                    "model_count": aggregate.model_count if aggregate else 0,
                    "models": ";".join(call.model_id for call in record.model_calls),
                    "input_tokens": sum(call.input_tokens for call in record.model_calls),
                    "output_tokens": sum(call.output_tokens for call in record.model_calls),
                    "cost_usd": f"{sum(call.cost_usd or 0 for call in record.model_calls):.8f}",
                    "budget_charged_usd": f"{record.budget_charged_usd:.8f}",
                    "prompt_version": record.prompt_version,
                    "error": record.error or "",
                }
            )
    temporary.replace(csv_path)
