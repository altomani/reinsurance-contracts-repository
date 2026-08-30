"""Command-line interface for dry runs, pilots, and gated corpus runs."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .extraction import RequestLimits
from .metadata import load_metadata_records
from .provider import OpenRouterClassifier, resolve_routes
from .runner import RunnerConfig, run_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify EDGAR exhibits under the five-gate reinsurance policy."
    )
    parser.add_argument("--index-dir", type=Path, default=Path("index-download"))
    parser.add_argument("--download-dir", type=Path, default=Path("download"))
    parser.add_argument("--year", type=int, action="append", dest="years")
    parser.add_argument("--file", action="append", dest="filenames")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV selection containing download_filename and, optionally, split.",
    )
    parser.add_argument(
        "--split",
        help="When --manifest is used, select only rows with this split value.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--model",
        action="append",
        help="Ordered route alias or model ID; repeat for escalation routes.",
    )
    parser.add_argument(
        "--benchmark-all",
        action="store_true",
        help="Call every selected model for every supported document.",
    )
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-input-chars", type=int, default=24_000)
    parser.add_argument("--max-input-tokens", type=int, default=12_000)
    parser.add_argument("--max-output-tokens", type=int, default=2_500)
    parser.add_argument("--context-window-tokens", type=int, default=32_768)
    parser.add_argument("--context-safety-tokens", type=int, default=2_000)
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument("--request-cost-reserve-usd", type=float, default=0.05)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--prompt", type=Path, default=Path("prompts/classifier-v1.md"))
    parser.add_argument(
        "--output-jsonl", type=Path, default=Path("results/classification-audit.jsonl")
    )
    parser.add_argument(
        "--output-csv", type=Path, default=Path("results/classification-latest.csv")
    )
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument(
        "--allow-full-corpus",
        action="store_true",
        help="Acknowledge the quality/forecast gate for an unbounded paid run.",
    )
    parser.add_argument(
        "--gate-report",
        type=Path,
        help="Forecast gate JSON required for an unbounded paid run.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.split and not args.manifest:
        parser.error("--split requires --manifest")
    if args.manifest and args.filenames:
        parser.error("--manifest and --file cannot be combined")
    if args.manifest:
        try:
            args.filenames = _manifest_filenames(args.manifest, split=args.split)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    unbounded_paid_run = not args.dry_run and args.limit is None and not args.filenames
    if unbounded_paid_run and not args.allow_full_corpus:
        parser.error("an unbounded paid run requires --allow-full-corpus")
    try:
        routes = resolve_routes(args.model)
        prompt_text = args.prompt.read_text(encoding="utf-8")
        limits = RequestLimits(
            max_input_chars=args.max_input_chars,
            max_input_tokens=args.max_input_tokens,
            max_output_tokens=args.max_output_tokens,
            context_window_tokens=args.context_window_tokens,
            context_safety_tokens=args.context_safety_tokens,
        )
        if unbounded_paid_run:
            _validate_full_run_gate(
                args.gate_report,
                prompt_version=args.prompt.stem,
                cli_budget_usd=args.budget_usd,
            )
        records = list(
            load_metadata_records(
                args.index_dir,
                args.download_dir,
                years=args.years,
                filenames=args.filenames,
                limit=args.limit,
            )
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not records:
        parser.error("selection matched no metadata rows")

    backend = None
    if not args.dry_run:
        load_dotenv()
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            parser.error(f"{args.api_key_env} is not set")
        backend = OpenRouterClassifier(
            api_key,
            app_title="EDGAR reinsurance contract classifier",
        )
    config = RunnerConfig(
        prompt_text=prompt_text,
        prompt_version=args.prompt.stem,
        routes=routes,
        limits=limits,
        budget_usd=args.budget_usd,
        request_cost_reserve_usd=args.request_cost_reserve_usd,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
        benchmark_all=args.benchmark_all,
        resume=args.resume,
        jsonl_path=args.output_jsonl,
        csv_path=args.output_csv,
    )
    summary = asyncio.run(run_records(records, backend, config))
    print(
        json.dumps(
            {
                "counts": summary.counts,
                "spent_usd": round(summary.spent_usd, 8),
                "reserved_usd": round(summary.reserved_usd, 8),
                "skipped_by_resume": summary.skipped_by_resume,
                "jsonl": str(args.output_jsonl),
                "csv": str(args.output_csv),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _validate_full_run_gate(
    gate_path: Path | None, *, prompt_version: str, cli_budget_usd: float
) -> None:
    if gate_path is None:
        raise ValueError("an unbounded paid run requires --gate-report")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("full_run_permitted") is not True:
        raise ValueError("the forecast gate does not permit a full run")
    if gate.get("prompt_version") != prompt_version:
        raise ValueError("the gate report was produced for a different prompt version")
    gated_budget = gate.get("forecast", {}).get("cli_budget_usd")
    if not isinstance(gated_budget, int | float) or cli_budget_usd > gated_budget:
        raise ValueError("the requested CLI budget exceeds the gate report budget")


def _manifest_filenames(path: Path, *, split: str | None) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "download_filename" not in rows[0]:
        raise ValueError("manifest must contain download_filename")
    if split and "split" not in rows[0]:
        raise ValueError("manifest must contain split when --split is used")
    filenames = [
        row["download_filename"].strip()
        for row in rows
        if row["download_filename"].strip()
        and (not split or row.get("split", "").strip() == split)
    ]
    filenames = list(dict.fromkeys(filenames))
    if not filenames:
        raise ValueError("manifest selection is empty")
    return filenames


if __name__ == "__main__":
    main()
