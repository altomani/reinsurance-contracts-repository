"""Prepare, compare, and finalize leakage-free independent gold reviews."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from .extraction import normalize_file
from .models import (
    BusinessBasis,
    Certainty,
    Completeness,
    CounterpartyDisclosure,
    DocumentKind,
    GovernmentBasis,
    PlacementBasis,
    PoolInvolvement,
    PoolKind,
    RejectionReason,
    TermStatus,
    TriState,
)
from .sampling import GOLD_LABEL_FIELDS


IDENTITY_FIELDS = ("document_id", "year", "download_filename", "split")
DECISION_FIELDS = (
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
)
EVIDENCE_FIELDS = (
    "evidence_reinsurance",
    "evidence_completeness",
    "evidence_business",
    "evidence_placement",
    "evidence_government",
)
REVIEW_VALUE_FIELDS = DECISION_FIELDS + EVIDENCE_FIELDS
REVIEW_FIELDS = IDENTITY_FIELDS + ("reviewer",) + REVIEW_VALUE_FIELDS + (
    "reviewer_notes",
)
ADJUDICATION_FIELDS = (
    IDENTITY_FIELDS
    + (
        "reviewer_1",
        "reviewer_2",
        "disagreement_fields",
        "reviewer_1_notes",
        "reviewer_2_notes",
        "adjudicator",
        "adjudication_notes",
    )
    + tuple(
        name
        for field in REVIEW_VALUE_FIELDS
        for name in (f"reviewer_1_{field}", f"reviewer_2_{field}", f"final_{field}")
    )
)
PACKET_MANIFEST_FIELDS = (
    "document_id",
    "year",
    "download_filename",
    "split",
    "packet_filename",
    "source_format",
    "source_bytes",
    "normalized_lines",
    "normalized_chars",
)

_ENUM_VALUES = {
    "document_kind": {item.value for item in DocumentKind},
    "is_reinsurance_contract": {item.value for item in TriState},
    "relationship_term": {item.value for item in TermStatus},
    "business_covered_term": {item.value for item in TermStatus},
    "term_period_term": {item.value for item in TermStatus},
    "risk_economics_term": {item.value for item in TermStatus},
    "premium_term": {item.value for item in TermStatus},
    "overall_completeness": {item.value for item in Completeness},
    "business_basis": {item.value for item in BusinessBasis},
    "placement_basis": {item.value for item in PlacementBasis},
    "counterparty_disclosure": {item.value for item in CounterpartyDisclosure},
    "government_basis": {item.value for item in GovernmentBasis},
    "pool_involvement": {item.value for item in PoolInvolvement},
    "pool_kind": {item.value for item in PoolKind},
    "certainty": {item.value for item in Certainty},
    "primary_rejection_reason": {item.value for item in RejectionReason},
}
_OPTIONAL_FIELDS = {"pool_exact_name", "pool_jurisdiction_or_authority"}
_LINE_REFERENCE = re.compile(r"L(\d+)(?:-L?(\d+))?", re.I)


def prepare_review_sheet(
    candidates_path: Path, output_path: Path, *, reviewer_id: str
) -> None:
    reviewer_id = reviewer_id.strip()
    if not reviewer_id:
        raise ValueError("reviewer_id cannot be empty")
    candidates = _read_rows(candidates_path, required=IDENTITY_FIELDS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for row in candidates:
            writer.writerow(
                {
                    **{field: row[field] for field in IDENTITY_FIELDS},
                    "reviewer": reviewer_id,
                }
            )


def export_review_packet(
    candidates_path: Path,
    download_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    """Export complete normalized, numbered documents without selection leakage."""

    candidates = _read_rows(candidates_path, required=IDENTITY_FIELDS)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise ValueError(
            f"review packet directory is not empty: {output_dir}; use --overwrite explicitly"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str | int]] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(candidates, start=1):
        document_id = row["document_id"]
        if document_id in seen_ids:
            raise ValueError(f"duplicate candidate document_id: {document_id!r}")
        seen_ids.add(document_id)
        filename = row["download_filename"]
        if Path(filename).name != filename:
            raise ValueError(f"unsafe download filename: {filename!r}")
        source_path = download_dir / filename
        if not source_path.is_file():
            raise ValueError(f"candidate document does not exist: {source_path}")
        document = normalize_file(source_path)
        safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", document_id).strip("._")
        if not safe_id:
            safe_id = "document"
        packet_filename = f"{index:04d}-{safe_id[:113]}.review.txt"
        packet_path = output_dir / packet_filename
        if packet_path.exists() and not overwrite:
            raise ValueError(f"review packet file already exists: {packet_path}")
        header = (
            f"DOCUMENT ID: {document_id}\n"
            f"YEAR: {row['year']}\n"
            f"DOWNLOAD FILENAME: {filename}\n"
            f"FROZEN SPLIT: {row['split']}\n"
            f"SOURCE FORMAT: {document.source_format}\n"
            f"NORMALIZED LINES: {len(document.lines)}\n"
            "\n<normalized-document>\n"
        )
        body = "\n".join(
            f"[L{line_number:06d}] {line}"
            for line_number, line in enumerate(document.lines, start=1)
        )
        packet_path.write_text(
            header + body + "\n</normalized-document>\n", encoding="utf-8"
        )
        manifest_rows.append(
            {
                "document_id": document_id,
                "year": row["year"],
                "download_filename": filename,
                "split": row["split"],
                "packet_filename": packet_filename,
                "source_format": document.source_format,
                "source_bytes": document.stats.source_bytes,
                "normalized_lines": len(document.lines),
                "normalized_chars": document.stats.normalized_chars,
            }
        )
    manifest_path = output_dir / "review-packet-manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PACKET_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    return {
        "documents": len(manifest_rows),
        "normalized_characters": sum(
            int(row["normalized_chars"]) for row in manifest_rows
        ),
    }


def compare_reviews(
    review_one_path: Path,
    review_two_path: Path,
    output_path: Path,
    *,
    download_dir: Path,
) -> dict[str, int]:
    line_count_cache: dict[str, int] = {}
    first, reviewer_one = _load_submission(
        review_one_path, download_dir, line_count_cache
    )
    second, reviewer_two = _load_submission(
        review_two_path, download_dir, line_count_cache
    )
    if reviewer_one == reviewer_two:
        raise ValueError("the two review submissions must use distinct reviewer IDs")
    first_by_id = {row["document_id"]: row for row in first}
    second_by_id = {row["document_id"]: row for row in second}
    if first_by_id.keys() != second_by_id.keys():
        missing_first = sorted(second_by_id.keys() - first_by_id.keys())
        missing_second = sorted(first_by_id.keys() - second_by_id.keys())
        raise ValueError(
            "review submissions contain different document sets; "
            f"missing from first={missing_first[:5]}, missing from second={missing_second[:5]}"
        )

    output_rows: list[dict[str, str]] = []
    disagreement_documents = 0
    disagreement_fields = 0
    for document_id in first_by_id:
        left = first_by_id[document_id]
        right = second_by_id[document_id]
        for field in IDENTITY_FIELDS:
            if left[field] != right[field]:
                raise ValueError(f"identity mismatch for {document_id!r}: {field}")
        differences = [
            field for field in REVIEW_VALUE_FIELDS if left[field] != right[field]
        ]
        disagreement_documents += int(bool(differences))
        disagreement_fields += len(differences)
        output: dict[str, str] = {
            **{field: left[field] for field in IDENTITY_FIELDS},
            "reviewer_1": reviewer_one,
            "reviewer_2": reviewer_two,
            "disagreement_fields": ";".join(differences),
            "reviewer_1_notes": left.get("reviewer_notes", ""),
            "reviewer_2_notes": right.get("reviewer_notes", ""),
            "adjudicator": "",
            "adjudication_notes": "",
        }
        for field in REVIEW_VALUE_FIELDS:
            output[f"reviewer_1_{field}"] = left[field]
            output[f"reviewer_2_{field}"] = right[field]
            output[f"final_{field}"] = left[field] if left[field] == right[field] else ""
        output_rows.append(output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ADJUDICATION_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)
    return {
        "documents": len(output_rows),
        "disagreement_documents": disagreement_documents,
        "disagreement_fields": disagreement_fields,
    }


def finalize_adjudication(
    adjudication_path: Path,
    output_path: Path,
    *,
    download_dir: Path,
) -> dict[str, int]:
    rows = _read_rows(adjudication_path, required=ADJUDICATION_FIELDS)
    cache: dict[str, int] = {}
    finalized: list[dict[str, str]] = []
    seen: set[str] = set()
    disagreements = 0
    for row in rows:
        document_id = row["document_id"]
        if document_id in seen:
            raise ValueError(f"duplicate adjudication document_id: {document_id!r}")
        seen.add(document_id)
        identities = {
            row["reviewer_1"].strip(),
            row["reviewer_2"].strip(),
            row["adjudicator"].strip(),
        }
        if "" in identities or len(identities) != 3:
            raise ValueError(
                f"{document_id!r} requires two distinct reviewers and a distinct adjudicator"
            )
        difference_fields = [
            field for field in row.get("disagreement_fields", "").split(";") if field
        ]
        if difference_fields and not row.get("adjudication_notes", "").strip():
            raise ValueError(f"{document_id!r} has disagreements but no adjudication_notes")
        disagreements += int(bool(difference_fields))
        final_values = {
            field: row.get(f"final_{field}", "").strip()
            for field in REVIEW_VALUE_FIELDS
        }
        _validate_label_values(
            final_values,
            filename=row["download_filename"],
            download_dir=download_dir,
            line_count_cache=cache,
        )
        qualifies, expected_reason = derive_qualification(final_values)
        if final_values["primary_rejection_reason"] != expected_reason:
            raise ValueError(
                f"{document_id!r} primary_rejection_reason should be {expected_reason!r}"
            )
        finalized.append(
            {
                "document_id": document_id,
                "year": row["year"],
                "download_filename": row["download_filename"],
                "split": row["split"],
                "selection_stratum": "",
                "reviewer_1": row["reviewer_1"],
                "reviewer_2": row["reviewer_2"],
                "adjudicator": row["adjudicator"],
                **final_values,
                "qualifies": str(qualifies).lower(),
                "adjudication_notes": row.get("adjudication_notes", ""),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GOLD_LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(finalized)
    return {"documents": len(finalized), "disagreement_documents": disagreements}


def derive_qualification(values: dict[str, str]) -> tuple[bool, str]:
    passing_kinds = {
        DocumentKind.COMPLETE_CONTRACT.value,
        DocumentKind.NEARLY_COMPLETE_CONTRACT.value,
    }
    if (
        values["document_kind"] in passing_kinds
        and values["is_reinsurance_contract"] == TriState.YES.value
    ):
        document = "pass"
    elif (
        values["document_kind"] not in passing_kinds
        and values["document_kind"] != DocumentKind.UNCLEAR.value
    ) or values["is_reinsurance_contract"] == TriState.NO.value:
        document = "fail"
    else:
        document = "unclear"

    term_values = [
        values[field]
        for field in (
            "relationship_term",
            "business_covered_term",
            "term_period_term",
            "risk_economics_term",
            "premium_term",
        )
    ]
    if (
        values["overall_completeness"] == Completeness.SUFFICIENT.value
        and all(
            value in {TermStatus.PRESENT.value, TermStatus.REDACTED.value}
            for value in term_values
        )
    ):
        main_terms = "pass"
    elif (
        values["overall_completeness"] == Completeness.INSUFFICIENT.value
        or TermStatus.MISSING.value in term_values
    ):
        main_terms = "fail"
    else:
        main_terms = "unclear"
    business = {
        BusinessBasis.NON_LIFE.value: "pass",
        BusinessBasis.LIFE_LIKE.value: "fail",
        BusinessBasis.MIXED.value: "fail",
        BusinessBasis.UNCLEAR.value: "unclear",
    }[values["business_basis"]]
    placement = {
        PlacementBasis.TREATY.value: "pass",
        PlacementBasis.FACULTATIVE.value: "fail",
        PlacementBasis.MIXED.value: "fail",
        PlacementBasis.UNCLEAR.value: "unclear",
    }[values["placement_basis"]]
    government = {
        GovernmentBasis.PRIVATE_MARKET.value: "pass",
        GovernmentBasis.STATUTORY_GOVERNMENT_SCHEME.value: "fail",
        GovernmentBasis.UNCLEAR.value: "unclear",
    }[values["government_basis"]]
    gates = (document, main_terms, business, placement, government)
    qualifies = all(gate == "pass" for gate in gates)

    if document == "fail":
        if values["document_kind"] == DocumentKind.AMENDMENT_OR_ENDORSEMENT.value:
            reason = RejectionReason.AMENDMENT_OR_ENDORSEMENT.value
        elif values["document_kind"] == DocumentKind.PLACEMENT_SLIP_OR_SUMMARY.value:
            reason = RejectionReason.PLACEMENT_SLIP_OR_SUMMARY.value
        else:
            reason = RejectionReason.NOT_REINSURANCE_CONTRACT.value
    elif main_terms == "fail":
        reason = RejectionReason.INSUFFICIENT_MAIN_TERMS.value
    elif business == "fail":
        reason = (
            RejectionReason.MIXED_BUSINESS_OR_PLACEMENT.value
            if values["business_basis"] == BusinessBasis.MIXED.value
            else RejectionReason.LIFE_LIKE_BUSINESS.value
        )
    elif placement == "fail":
        reason = (
            RejectionReason.MIXED_BUSINESS_OR_PLACEMENT.value
            if values["placement_basis"] == PlacementBasis.MIXED.value
            else RejectionReason.FACULTATIVE_PLACEMENT.value
        )
    elif government == "fail":
        reason = RejectionReason.STATUTORY_GOVERNMENT_SCHEME.value
    elif "unclear" in gates:
        reason = RejectionReason.UNCLEAR_DECISIVE_CRITERION.value
    else:
        reason = RejectionReason.NONE.value
    return qualifies, reason


def _load_submission(
    path: Path, download_dir: Path, cache: dict[str, int]
) -> tuple[list[dict[str, str]], str]:
    rows = _read_rows(path, required=REVIEW_FIELDS)
    if not rows:
        raise ValueError(f"review submission is empty: {path}")
    reviewer_ids = {row.get("reviewer", "").strip() for row in rows}
    if "" in reviewer_ids or len(reviewer_ids) != 1:
        raise ValueError(f"review submission must contain exactly one reviewer ID: {path}")
    seen: set[str] = set()
    for row in rows:
        document_id = row["document_id"]
        if document_id in seen:
            raise ValueError(f"duplicate review document_id: {document_id!r}")
        seen.add(document_id)
        _validate_label_values(
            row,
            filename=row["download_filename"],
            download_dir=download_dir,
            line_count_cache=cache,
        )
        _, expected_reason = derive_qualification(row)
        if row["primary_rejection_reason"] != expected_reason:
            raise ValueError(
                f"{document_id!r} primary_rejection_reason should be {expected_reason!r}"
            )
    return rows, reviewer_ids.pop()


def _validate_label_values(
    values: dict[str, str],
    *,
    filename: str,
    download_dir: Path,
    line_count_cache: dict[str, int],
) -> None:
    for field in DECISION_FIELDS:
        value = values.get(field, "").strip()
        if field not in _OPTIONAL_FIELDS and not value:
            raise ValueError(f"{filename!r} is missing required field {field!r}")
        if field in _ENUM_VALUES and value not in _ENUM_VALUES[field]:
            raise ValueError(f"{filename!r} has invalid {field}={value!r}")
    if (
        values.get("pool_involvement") == PoolInvolvement.NONE.value
        and values.get("pool_exact_name", "").strip()
    ):
        raise ValueError(f"{filename!r} names a pool while pool_involvement is none")
    line_count = _document_line_count(filename, download_dir, line_count_cache)
    for field in EVIDENCE_FIELDS:
        value = values.get(field, "").strip()
        matches = list(_LINE_REFERENCE.finditer(value))
        if not value or not matches:
            raise ValueError(f"{filename!r} field {field!r} needs numbered-line evidence")
        for match in matches:
            start = int(match.group(1))
            end = int(match.group(2) or start)
            if end < start or start < 1 or end > line_count:
                raise ValueError(
                    f"{filename!r} field {field!r} references lines outside 1-{line_count}"
                )


def _document_line_count(
    filename: str, download_dir: Path, cache: dict[str, int]
) -> int:
    if filename not in cache:
        if Path(filename).name != filename:
            raise ValueError(f"unsafe download filename: {filename!r}")
        path = download_dir / filename
        if not path.is_file():
            raise ValueError(f"reviewed document does not exist: {path}")
        cache[filename] = len(normalize_file(path).lines)
    return cache[filename]


def _read_rows(path: Path, *, required: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = [field for field in required if field not in fields]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        return [
            {key: (value or "").strip() for key, value in row.items() if key is not None}
            for row in reader
        ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create one blinded review sheet.")
    prepare.add_argument("--candidates", type=Path, required=True)
    prepare.add_argument("--reviewer-id", required=True)
    prepare.add_argument("--output", type=Path, required=True)

    export = subparsers.add_parser(
        "export", help="Export full normalized, line-numbered review documents."
    )
    export.add_argument("--candidates", type=Path, required=True)
    export.add_argument("--download-dir", type=Path, default=Path("download"))
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--overwrite", action="store_true")

    compare = subparsers.add_parser(
        "compare", help="Validate and compare two independent review submissions."
    )
    compare.add_argument("--review-one", type=Path, required=True)
    compare.add_argument("--review-two", type=Path, required=True)
    compare.add_argument("--download-dir", type=Path, default=Path("download"))
    compare.add_argument("--output", type=Path, required=True)

    finalize = subparsers.add_parser(
        "finalize", help="Validate an adjudication sheet and emit benchmark gold."
    )
    finalize.add_argument("--adjudication", type=Path, required=True)
    finalize.add_argument("--download-dir", type=Path, default=Path("download"))
    finalize.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare_review_sheet(
                args.candidates, args.output, reviewer_id=args.reviewer_id
            )
            print(f"Prepared review sheet: {args.output}")
        elif args.command == "export":
            summary = export_review_packet(
                args.candidates,
                args.download_dir,
                args.output_dir,
                overwrite=args.overwrite,
            )
            print(
                f"Exported {summary['documents']} normalized review documents "
                f"({summary['normalized_characters']} characters)."
            )
        elif args.command == "compare":
            summary = compare_reviews(
                args.review_one,
                args.review_two,
                args.output,
                download_dir=args.download_dir,
            )
            print(
                f"Compared {summary['documents']} documents; "
                f"{summary['disagreement_documents']} need adjudication."
            )
        else:
            summary = finalize_adjudication(
                args.adjudication,
                args.output,
                download_dir=args.download_dir,
            )
            print(
                f"Finalized {summary['documents']} adjudicated documents "
                f"({summary['disagreement_documents']} with disagreements)."
            )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
