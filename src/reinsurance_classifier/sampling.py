"""Build a reproducible, stratified gold-label candidate manifest."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_QUOTAS = {
    "historical_disagreement": 32,
    "government_scheme": 16,
    "amendment_or_endorsement": 20,
    "placement_slip_or_summary": 16,
    "life_health_or_annuity": 20,
    "facultative_edge": 16,
    "clear_reinsurance_candidate": 28,
    "unrelated_agreement": 20,
    "other": 12,
}

MANIFEST_FIELDS = (
    "document_id",
    "year",
    "download_filename",
    "selection_stratum",
    "split",
    "selection_reasons",
    "description",
    "source_format",
    "source_bytes",
    "length_bin",
    "gpt_reinsurance",
    "gemini_reinsurance",
    "gpt_contract_type",
    "gemini_contract_type",
    "gpt_obligatory_type",
    "gemini_obligatory_type",
)

GOLD_LABEL_FIELDS = (
    "document_id",
    "year",
    "download_filename",
    "split",
    "selection_stratum",
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
    "counterparty_disclosure",
    "government_basis",
    "pool_involvement",
    "pool_kind",
    "pool_exact_name",
    "pool_jurisdiction_or_authority",
    "certainty",
    "primary_rejection_reason",
    "qualifies",
    "evidence_reinsurance",
    "evidence_completeness",
    "evidence_business",
    "evidence_placement",
    "evidence_government",
    "adjudication_notes",
)


def _read_rows(paths: list[Path]) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            for row in csv.DictReader(handle):
                filename = (row.get("downloadFilename") or "").strip()
                if filename:
                    rows[filename] = {key: value or "" for key, value in row.items()}
    return rows


def _historical_disagreement(
    gpt: dict[str, str] | None, gemini: dict[str, str] | None
) -> list[str]:
    if not gpt or not gemini:
        return []
    reasons: list[str] = []
    for field in ("reinsurance", "contractType", "obligatoryType", "classOfBusiness"):
        left = gpt.get(field, "").strip().lower()
        right = gemini.get(field, "").strip().lower()
        if left and right and left != right:
            reasons.append(f"historical_{field}_disagreement")
    return reasons


def _description_reasons(description: str) -> list[str]:
    text = description.lower()
    patterns = {
        "government_scheme": r"hurricane catastrophe fund|reimbursement contract|statutory (?:fund|pool)|government reinsurance",
        "amendment_or_endorsement": r"\bamend(?:ment|ed)?\b|\bendorse(?:ment)?\b|\bextension\b|\bcommutation\b",
        "placement_slip_or_summary": r"placement slip|cover note|\bbinder\b|term sheet|reinsurance confirmation",
        "life_health_or_annuity": r"\blife\b|annuit|pension|longevity|health plan|employee benefit|coinsurance",
        "facultative_edge": r"\bfacultative\b|fac[- ]oblig",
        "clear_reinsurance_candidate": r"reinsurance (?:contract|agreement|treaty)|retrocession|excess of loss|quota share",
        "unrelated_agreement": r"credit agreement|stock purchase|settlement agreement|employment agreement|merger agreement|lease agreement|services agreement",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text)]


def _weak_label_reasons(
    gpt: dict[str, str] | None, gemini: dict[str, str] | None
) -> list[str]:
    rows = [row for row in (gpt, gemini) if row]
    reasons: list[str] = []
    if any(row.get("obligatoryType", "").lower() == "facultative" for row in rows):
        reasons.append("facultative_edge")
    if any(
        row.get("contractType", "").lower() == "life"
        or row.get("classOfBusiness", "").lower() in {"life", "health"}
        for row in rows
    ):
        reasons.append("life_health_or_annuity")
    if any(row.get("reinsurance", "").lower() == "yes" for row in rows):
        reasons.append("clear_reinsurance_candidate")
    return reasons


def _primary_stratum(reasons: list[str]) -> str:
    # Preserve scarce substantive edge cases before assigning the broader
    # historical-disagreement stratum.
    for stratum in (
        "government_scheme",
        "amendment_or_endorsement",
        "placement_slip_or_summary",
        "life_health_or_annuity",
        "facultative_edge",
    ):
        if stratum in reasons:
            return stratum
    if any(reason.startswith("historical_") for reason in reasons):
        return "historical_disagreement"
    for stratum in ("clear_reinsurance_candidate", "unrelated_agreement"):
        if stratum in reasons:
            return stratum
    return "other"


def _length_bin(size: int) -> str:
    if size < 50_000:
        return "lt_50kb"
    if size < 250_000:
        return "50_250kb"
    if size < 1_000_000:
        return "250kb_1mb"
    return "gte_1mb"


def build_candidate_manifest(
    *,
    index_dir: Path,
    download_dir: Path,
    gpt_dir: Path,
    gemini_dir: Path,
    target: int = 180,
    seed: int = 20260829,
    exclude_filenames: set[str] | None = None,
) -> list[dict[str, str | int]]:
    if target < 1:
        raise ValueError("target must be positive")
    metadata = _read_rows(sorted(index_dir.glob("index-*.csv")))
    gpt = _read_rows(sorted(gpt_dir.glob("*.csv")))
    gemini = _read_rows(sorted(gemini_dir.glob("*.csv")))
    groups: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    excluded = exclude_filenames or set()
    for filename, row in metadata.items():
        if filename in excluded:
            continue
        path = download_dir / filename
        if not path.is_file() or path.suffix.lower() == ".pdf":
            continue
        gpt_row = gpt.get(filename, {})
        gemini_row = gemini.get(filename, {})
        disagreement = _historical_disagreement(gpt_row, gemini_row)
        reasons = (
            disagreement
            + _description_reasons(row.get("description", ""))
            + _weak_label_reasons(gpt_row, gemini_row)
        )
        reasons = list(dict.fromkeys(reasons))
        stratum = _primary_stratum(reasons)
        size = path.stat().st_size
        try:
            year = int(filename[:4])
        except ValueError:
            year = 0
        groups[stratum].append(
            {
                "document_id": filename,
                "year": year,
                "download_filename": filename,
                "selection_stratum": stratum,
                "selection_reasons": ";".join(reasons or ["random_other"]),
                "description": row.get("description", ""),
                "source_format": path.suffix.lower().lstrip("."),
                "source_bytes": size,
                "length_bin": _length_bin(size),
                "gpt_reinsurance": gpt_row.get("reinsurance", ""),
                "gemini_reinsurance": gemini_row.get("reinsurance", ""),
                "gpt_contract_type": gpt_row.get("contractType", ""),
                "gemini_contract_type": gemini_row.get("contractType", ""),
                "gpt_obligatory_type": gpt_row.get("obligatoryType", ""),
                "gemini_obligatory_type": gemini_row.get("obligatoryType", ""),
            }
        )

    rng = random.Random(seed)
    selected: list[dict[str, str | int]] = []
    leftovers: list[dict[str, str | int]] = []
    scale = target / sum(DEFAULT_QUOTAS.values())
    quotas = {
        stratum: max(1, round(quota * scale))
        for stratum, quota in DEFAULT_QUOTAS.items()
    }
    while sum(quotas.values()) > target:
        largest = max(quotas, key=quotas.get)
        quotas[largest] -= 1
    while sum(quotas.values()) < target:
        smallest = min(quotas, key=quotas.get)
        quotas[smallest] += 1

    for stratum in DEFAULT_QUOTAS:
        candidates = groups.get(stratum, [])
        rng.shuffle(candidates)
        # Sorting on a random key keeps selection deterministic while avoiding a
        # year-ordered sample; quotas still preserve the edge-case strata.
        candidates.sort(key=lambda _: rng.random())
        quota = quotas[stratum]
        selected.extend(candidates[:quota])
        leftovers.extend(candidates[quota:])
    if len(selected) < target:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target - len(selected)])
    selected = selected[:target]
    by_stratum: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    for row in selected:
        by_stratum[str(row["selection_stratum"])].append(row)
    for rows in by_stratum.values():
        rng.shuffle(rows)
        holdout_count = max(1, round(len(rows) / 3))
        for index, row in enumerate(rows):
            row["split"] = "holdout" if index < holdout_count else "development"
    selected.sort(key=lambda row: (int(row["year"]), str(row["download_filename"])))
    return selected


def write_manifest(rows: list[dict[str, str | int]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_blinded_label_sheet(
    rows: list[dict[str, str | int]], output: Path
) -> None:
    """Give reviewers IDs and frozen splits without weak-label or stratum leakage."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GOLD_LABEL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "document_id": row["document_id"],
                    "year": row["year"],
                    "download_filename": row["download_filename"],
                    "split": row["split"],
                    "selection_stratum": "",
                }
            )


def manifest_summary(rows: list[dict[str, str | int]]) -> dict:
    return {
        "documents": len(rows),
        "strata": dict(sorted(Counter(row["selection_stratum"] for row in rows).items())),
        "years": dict(sorted(Counter(row["year"] for row in rows).items())),
        "formats": dict(sorted(Counter(row["source_format"] for row in rows).items())),
        "length_bins": dict(sorted(Counter(row["length_bin"] for row in rows).items())),
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path("index-download"))
    parser.add_argument("--download-dir", type=Path, default=Path("download"))
    parser.add_argument(
        "--gpt-dir",
        type=Path,
        default=Path("archive/classifiers/results/gpt-4o-mini"),
    )
    parser.add_argument(
        "--gemini-dir",
        type=Path,
        default=Path("archive/classifiers/results/gemini-2.0-flash"),
    )
    parser.add_argument("--target", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        help="Exclude download_filename values already used by another manifest.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("gold/candidate-manifest.csv")
    )
    parser.add_argument(
        "--label-output", type=Path, default=Path("gold/candidate-labeling.csv")
    )
    args = parser.parse_args(argv)
    excluded: set[str] = set()
    if args.exclude_manifest:
        with args.exclude_manifest.open(newline="", encoding="utf-8-sig") as handle:
            excluded = {
                row.get("download_filename", "").strip()
                for row in csv.DictReader(handle)
                if row.get("download_filename", "").strip()
            }
    rows = build_candidate_manifest(
        index_dir=args.index_dir,
        download_dir=args.download_dir,
        gpt_dir=args.gpt_dir,
        gemini_dir=args.gemini_dir,
        target=args.target,
        seed=args.seed,
        exclude_filenames=excluded,
    )
    write_manifest(rows, args.output)
    write_blinded_label_sheet(rows, args.label_output)
    print(json.dumps(manifest_summary(rows), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
