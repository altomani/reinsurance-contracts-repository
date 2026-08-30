"""Resumable classification orchestration with bounded retries and routing."""

from __future__ import annotations

import asyncio
import hashlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .aggregation import (
    AggregatedClassification,
    OutcomeStatus,
    aggregate_decisions,
    criteria_disagree,
    has_rule_contradiction,
    needs_second_model,
)
from .budget import BudgetExceeded, BudgetLedger
from .extraction import EvidencePack, RequestLimits, build_evidence_pack, normalize_file
from .metadata import MetadataRecord, SourceStatus
from .models import RejectionReason
from .output import (
    AuditRecord,
    AuditWriter,
    EvidencePackAudit,
    RecordStatus,
    TERMINAL_STATUSES,
    read_audit_records,
    read_latest_records,
    recorded_cost,
    write_latest_csv,
)
from .provider import ClassifierBackend, ModelRoute, ProviderCallResult


@dataclass(frozen=True)
class RunnerConfig:
    prompt_text: str
    prompt_version: str
    routes: tuple[ModelRoute, ...]
    limits: RequestLimits
    budget_usd: float
    request_cost_reserve_usd: float
    concurrency: int
    max_retries: int
    dry_run: bool
    benchmark_all: bool
    resume: bool
    jsonl_path: Path
    csv_path: Path

    def __post_init__(self) -> None:
        if not self.routes and not self.dry_run:
            raise ValueError("at least one model route is required")
        if not 1 <= self.concurrency <= 32:
            raise ValueError("concurrency must be between 1 and 32")
        if not 0 <= self.max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        if self.request_cost_reserve_usd <= 0:
            raise ValueError("request cost reserve must be positive")


@dataclass(frozen=True)
class RunSummary:
    counts: dict[str, int]
    spent_usd: float
    reserved_usd: float
    skipped_by_resume: int


@dataclass(frozen=True)
class _ChargedCall:
    result: ProviderCallResult
    budget_charged_usd: float


class _CallFailure(RuntimeError):
    def __init__(self, cause: Exception, budget_charged_usd: float) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.budget_charged_usd = budget_charged_usd


class _RoutingFailure(RuntimeError):
    def __init__(
        self,
        cause: Exception,
        calls: list[ProviderCallResult],
        budget_charged_usd: float,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.calls = calls
        self.budget_charged_usd = budget_charged_usd


async def run_records(
    records: list[MetadataRecord],
    backend: ClassifierBackend | None,
    config: RunnerConfig,
) -> RunSummary:
    latest = read_latest_records(config.jsonl_path) if config.resume else {}
    completed_ids = {
        record_id
        for record_id, record in latest.items()
        if record.status in TERMINAL_STATUSES
    }
    if config.dry_run:
        completed_ids.update(
            record_id
            for record_id, record in latest.items()
            if record.status == RecordStatus.PREPARED
        )
    initial_spent = (
        recorded_cost(read_audit_records(config.jsonl_path)) if config.resume else 0.0
    )
    ledger = BudgetLedger(config.budget_usd, initial_spent_usd=initial_spent)
    writer = AuditWriter(config.jsonl_path)
    semaphore = asyncio.Semaphore(config.concurrency)
    process_pool = (
        ProcessPoolExecutor(max_workers=config.concurrency) if config.dry_run else None
    )
    counts: Counter[str] = Counter()
    skipped = 0

    async def process(record: MetadataRecord) -> None:
        nonlocal skipped
        if record.record_id in completed_ids:
            skipped += 1
            return
        async with semaphore:
            if process_pool is not None:
                audit = await asyncio.get_running_loop().run_in_executor(
                    process_pool,
                    _prepare_dry_record,
                    record,
                    config.prompt_text,
                    config.prompt_version,
                    config.limits,
                )
            else:
                audit = await _process_record(record, backend, config, ledger)
            await writer.append(audit)
            counts[audit.status.value] += 1

    try:
        await asyncio.gather(*(process(record) for record in records))
    finally:
        if process_pool is not None:
            process_pool.shutdown(wait=True, cancel_futures=True)
    write_latest_csv(config.jsonl_path, config.csv_path)
    snapshot = await ledger.snapshot()
    return RunSummary(
        counts=dict(sorted(counts.items())),
        spent_usd=snapshot.spent_usd,
        reserved_usd=snapshot.reserved_usd,
        skipped_by_resume=skipped,
    )


def _prepare_dry_record(
    record: MetadataRecord,
    prompt_text: str,
    prompt_version: str,
    limits: RequestLimits,
) -> AuditRecord:
    """Process-local preparation path used for corpus-scale no-API runs."""

    base = {
        "record_id": record.record_id,
        "year": record.year,
        "download_filename": record.download_filename,
        "prompt_version": prompt_version,
        "created_at": AuditRecord.now(),
        "metadata": record.metadata,
    }
    source_status_map = {
        SourceStatus.MISSING_FILE: RecordStatus.MISSING_FILE,
        SourceStatus.SKIPPED_PDF: RecordStatus.SKIPPED_PDF,
        SourceStatus.UNSUPPORTED_FILE: RecordStatus.UNSUPPORTED_FILE,
    }
    if record.source_status != SourceStatus.SUPPORTED:
        return AuditRecord(status=source_status_map[record.source_status], **base)
    assert record.source_path is not None
    try:
        source_hash = _source_sha256(record.source_path)
        document = normalize_file(record.source_path)
        pack = build_evidence_pack(
            document,
            record.metadata,
            limits=limits,
            prompt_text=prompt_text,
        )
        return AuditRecord(
            status=RecordStatus.PREPARED,
            source_sha256=source_hash,
            evidence_pack=EvidencePackAudit(
                truncated=pack.truncated,
                normalized_chars=pack.normalized_chars,
                selected_line_count=len(pack.selected_line_numbers),
                selected_ranges=list(pack.selected_ranges),
                categories_found=list(pack.categories_found),
                request_chars=len(pack.text),
                estimated_input_tokens=pack.estimated_input_tokens,
            ),
            **base,
        )
    except Exception as exc:
        return AuditRecord(
            status=RecordStatus.ERROR,
            error=f"{type(exc).__name__}: {exc}",
            **base,
        )


async def _process_record(
    record: MetadataRecord,
    backend: ClassifierBackend | None,
    config: RunnerConfig,
    ledger: BudgetLedger,
) -> AuditRecord:
    base = {
        "record_id": record.record_id,
        "year": record.year,
        "download_filename": record.download_filename,
        "prompt_version": config.prompt_version,
        "created_at": AuditRecord.now(),
        "metadata": record.metadata,
    }
    source_status_map = {
        SourceStatus.MISSING_FILE: RecordStatus.MISSING_FILE,
        SourceStatus.SKIPPED_PDF: RecordStatus.SKIPPED_PDF,
        SourceStatus.UNSUPPORTED_FILE: RecordStatus.UNSUPPORTED_FILE,
    }
    if record.source_status != SourceStatus.SUPPORTED:
        return AuditRecord(status=source_status_map[record.source_status], **base)
    assert record.source_path is not None
    try:
        source_hash = await asyncio.to_thread(_source_sha256, record.source_path)
        document = await asyncio.to_thread(normalize_file, record.source_path)
        pack = await asyncio.to_thread(
            build_evidence_pack,
            document,
            record.metadata,
            limits=config.limits,
            prompt_text=config.prompt_text,
        )
        pack_audit = EvidencePackAudit(
            truncated=pack.truncated,
            normalized_chars=pack.normalized_chars,
            selected_line_count=len(pack.selected_line_numbers),
            selected_ranges=list(pack.selected_ranges),
            categories_found=list(pack.categories_found),
            request_chars=len(pack.text),
            estimated_input_tokens=pack.estimated_input_tokens,
        )
        if config.dry_run:
            return AuditRecord(
                status=RecordStatus.PREPARED,
                source_sha256=source_hash,
                evidence_pack=pack_audit,
                **base,
            )
        if backend is None:
            raise RuntimeError("a classifier backend is required outside dry-run mode")
        calls, budget_charged = await _route_calls(pack, backend, config, ledger)
        aggregate = enforce_direct_evidence_gate(
            aggregate_decisions([call.decision for call in calls]), calls
        )
        status = {
            OutcomeStatus.QUALIFIED: RecordStatus.QUALIFIED,
            OutcomeStatus.REJECTED: RecordStatus.REJECTED,
            OutcomeStatus.MANUAL_REVIEW: RecordStatus.MANUAL_REVIEW,
        }[aggregate.status]
        return AuditRecord(
            status=status,
            source_sha256=source_hash,
            evidence_pack=pack_audit,
            model_calls=calls,
            aggregate=aggregate,
            budget_charged_usd=budget_charged,
            **base,
        )
    except _RoutingFailure as exc:
        status = (
            RecordStatus.BUDGET_EXHAUSTED
            if isinstance(exc.cause, BudgetExceeded)
            else RecordStatus.ERROR
        )
        return AuditRecord(
            status=status,
            source_sha256=locals().get("source_hash"),
            evidence_pack=locals().get("pack_audit"),
            model_calls=exc.calls,
            budget_charged_usd=exc.budget_charged_usd,
            error=f"{type(exc.cause).__name__}: {exc.cause}",
            **base,
        )
    except BudgetExceeded as exc:
        return AuditRecord(status=RecordStatus.BUDGET_EXHAUSTED, error=str(exc), **base)
    except Exception as exc:
        return AuditRecord(
            status=RecordStatus.ERROR,
            error=f"{type(exc).__name__}: {exc}",
            **base,
        )


async def _route_calls(
    pack: EvidencePack,
    backend: ClassifierBackend,
    config: RunnerConfig,
    ledger: BudgetLedger,
) -> tuple[list[ProviderCallResult], float]:
    calls: list[ProviderCallResult] = []
    budget_charged = 0.0

    async def call(route: ModelRoute) -> None:
        nonlocal budget_charged
        try:
            charged_call = await _call_with_retry(pack, route, backend, config, ledger)
        except _CallFailure as exc:
            budget_charged += exc.budget_charged_usd
            raise _RoutingFailure(exc.cause, calls, budget_charged) from exc.cause
        calls.append(charged_call.result)
        budget_charged += charged_call.budget_charged_usd

    if config.benchmark_all:
        for route in config.routes:
            await call(route)
        return calls, budget_charged
    await call(config.routes[0])
    first_qualifies = aggregate_decisions([calls[0].decision]).qualifies
    if len(config.routes) >= 2 and (
        needs_second_model(calls[0].decision) or first_qualifies
    ):
        await call(config.routes[1])
    if (
        len(config.routes) >= 3
        and len(calls) == 2
        and criteria_disagree(calls[0].decision, calls[1].decision)
    ):
        await call(config.routes[2])
    return calls, budget_charged


def enforce_direct_evidence_gate(
    aggregate: AggregatedClassification,
    calls: list[ProviderCallResult],
) -> AggregatedClassification:
    """Require one internally consistent provider decision with grounded evidence."""

    direct_evidence_valid = any(
        (call.evidence_lines_valid or call.evidence_quotes_valid)
        and not has_rule_contradiction(call.decision)
        for call in calls
    )
    if aggregate.qualifies and not direct_evidence_valid:
        return aggregate.model_copy(
            update={
                "qualifies": False,
                "status": OutcomeStatus.MANUAL_REVIEW,
                "primary_rejection_reason": (
                    RejectionReason.UNCLEAR_DECISIVE_CRITERION
                ),
                "direct_evidence_valid": False,
            }
        )
    return aggregate.model_copy(
        update={"direct_evidence_valid": direct_evidence_valid}
    )


async def _call_with_retry(
    pack: EvidencePack,
    route: ModelRoute,
    backend: ClassifierBackend,
    config: RunnerConfig,
    ledger: BudgetLedger,
) -> _ChargedCall:
    last_error: Exception | None = None
    total_charged = 0.0
    for attempt in range(config.max_retries + 1):
        try:
            reservation = await ledger.reserve(config.request_cost_reserve_usd)
        except BudgetExceeded as exc:
            raise _CallFailure(exc, total_charged) from exc
        try:
            result = await backend.classify(
                pack,
                route,
                prompt_text=config.prompt_text,
                max_output_tokens=config.limits.max_output_tokens,
            )
        except Exception as exc:
            total_charged += await ledger.settle(reservation, None)
            last_error = exc
            if attempt >= config.max_retries or not _is_transient(exc):
                raise _CallFailure(exc, total_charged) from exc
            await asyncio.sleep(min(2**attempt, 8))
            continue
        charged = await ledger.settle(reservation, result.cost_usd)
        total_charged += charged
        updates: dict[str, object] = {"retry_count": result.retry_count + attempt}
        if result.cost_usd is None:
            updates["cost_usd"] = charged
        return _ChargedCall(
            result=result.model_copy(update=updates),
            budget_charged_usd=total_charged,
        )
    assert last_error is not None
    raise _CallFailure(last_error, total_charged)


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in ("429", "rate limit", "timeout", "temporar", "connection", " 500", " 502", " 503", " 504")
    )


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
