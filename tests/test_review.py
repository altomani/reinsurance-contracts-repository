from __future__ import annotations

import csv
from pathlib import Path

import pytest

from reinsurance_classifier.review import (
    _validate_label_values,
    compare_reviews,
    derive_qualification,
    export_review_packet,
    finalize_adjudication,
    prepare_review_sheet,
)


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _positive_values() -> dict[str, str]:
    values = {
        "document_kind": "complete_contract",
        "is_reinsurance_contract": "yes",
        "relationship_term": "present",
        "business_covered_term": "present",
        "term_period_term": "present",
        "risk_economics_term": "present",
        "premium_term": "redacted",
        "overall_completeness": "sufficient",
        "business_basis": "non_life",
        "placement_basis": "treaty",
        "counterparty_disclosure": "named",
        "government_basis": "private_market",
        "pool_involvement": "none",
        "pool_kind": "other",
        "pool_exact_name": "",
        "pool_jurisdiction_or_authority": "",
        "certainty": "high",
        "primary_rejection_reason": "none",
    }
    for field in (
        "evidence_reinsurance",
        "evidence_completeness",
        "evidence_business",
        "evidence_placement",
        "evidence_government",
    ):
        values[field] = "L1: operative evidence"
    return values


def _candidate_sheet(path: Path, filenames: list[str]) -> None:
    fields = ["document_id", "year", "download_filename", "split"]
    rows = [
        {
            "document_id": f"doc-{index}",
            "year": "2024",
            "download_filename": filename,
            "split": "development" if index == 1 else "holdout",
        }
        for index, filename in enumerate(filenames, start=1)
    ]
    _write(path, fields, rows)


def test_independent_reviews_compare_and_finalize_to_derived_gold(tmp_path: Path) -> None:
    download = tmp_path / "download"
    download.mkdir()
    filenames = ["one.txt", "two.txt"]
    for filename in filenames:
        (download / filename).write_text("operative evidence", encoding="utf-8")
    candidates = tmp_path / "candidates.csv"
    _candidate_sheet(candidates, filenames)
    review_a = tmp_path / "review-a.csv"
    review_b = tmp_path / "review-b.csv"
    prepare_review_sheet(candidates, review_a, reviewer_id="reviewer-a")
    prepare_review_sheet(candidates, review_b, reviewer_id="reviewer-b")

    for path in (review_a, review_b):
        fields, rows = _read(path)
        for row in rows:
            row.update(_positive_values())
        if path == review_b:
            rows[1]["business_basis"] = "life_like"
            rows[1]["primary_rejection_reason"] = "life_like_business"
        _write(path, fields, rows)

    adjudication = tmp_path / "adjudication.csv"
    summary = compare_reviews(
        review_a, review_b, adjudication, download_dir=download
    )
    assert summary == {
        "documents": 2,
        "disagreement_documents": 1,
        "disagreement_fields": 2,
    }
    fields, rows = _read(adjudication)
    assert rows[0]["disagreement_fields"] == ""
    assert rows[0]["final_business_basis"] == "non_life"
    assert set(rows[1]["disagreement_fields"].split(";")) == {
        "business_basis",
        "primary_rejection_reason",
    }
    assert rows[1]["final_business_basis"] == ""

    for row in rows:
        row["adjudicator"] = "adjudicator-c"
    rows[1]["final_business_basis"] = "life_like"
    rows[1]["final_primary_rejection_reason"] = "life_like_business"
    rows[1]["adjudication_notes"] = "The annuity wording controls."
    _write(adjudication, fields, rows)

    gold = tmp_path / "adjudicated.csv"
    final_summary = finalize_adjudication(
        adjudication, gold, download_dir=download
    )
    assert final_summary == {"documents": 2, "disagreement_documents": 1}
    _, gold_rows = _read(gold)
    assert [row["qualifies"] for row in gold_rows] == ["true", "false"]
    assert all(row["reviewer_1"] == "reviewer-a" for row in gold_rows)
    assert all(row["reviewer_2"] == "reviewer-b" for row in gold_rows)
    assert all(row["adjudicator"] == "adjudicator-c" for row in gold_rows)


def test_review_packet_exports_complete_normalized_numbered_text(tmp_path: Path) -> None:
    download = tmp_path / "download"
    download.mkdir()
    (download / "one.htm").write_text(
        "<html><script>noise</script><h1>Agreement</h1><p>Premium term</p></html>",
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates.csv"
    _candidate_sheet(candidates, ["one.htm"])
    output = tmp_path / "packet"

    summary = export_review_packet(candidates, download, output)

    assert summary["documents"] == 1
    manifest_fields, manifest = _read(output / "review-packet-manifest.csv")
    assert "selection_stratum" not in manifest_fields
    packet = (output / manifest[0]["packet_filename"]).read_text(encoding="utf-8")
    assert "[L000001] Agreement" in packet
    assert "[L000002] Premium term" in packet
    assert "noise" not in packet
    with pytest.raises(ValueError, match="not empty"):
        export_review_packet(candidates, download, output)


def test_review_comparison_rejects_same_reviewer_and_bad_evidence(tmp_path: Path) -> None:
    download = tmp_path / "download"
    download.mkdir()
    (download / "one.txt").write_text("one line", encoding="utf-8")
    candidates = tmp_path / "candidates.csv"
    _candidate_sheet(candidates, ["one.txt"])
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    prepare_review_sheet(candidates, first, reviewer_id="same")
    prepare_review_sheet(candidates, second, reviewer_id="same")
    for path in (first, second):
        fields, rows = _read(path)
        rows[0].update(_positive_values())
        _write(path, fields, rows)

    with pytest.raises(ValueError, match="distinct reviewer"):
        compare_reviews(first, second, tmp_path / "out.csv", download_dir=download)

    fields, rows = _read(second)
    rows[0]["reviewer"] = "different"
    rows[0]["evidence_business"] = "L2: nonexistent"
    _write(second, fields, rows)
    with pytest.raises(ValueError, match="outside 1-1"):
        compare_reviews(first, second, tmp_path / "out.csv", download_dir=download)


def test_qualification_derivation_prioritizes_decisive_failure_over_unclear() -> None:
    values = _positive_values()
    values["document_kind"] = "amendment_or_endorsement"
    values["business_basis"] = "unclear"

    qualifies, reason = derive_qualification(values)

    assert qualifies is False
    assert reason == "amendment_or_endorsement"

    values = _positive_values()
    values["document_kind"] = "unclear"
    values["is_reinsurance_contract"] = "no"
    assert derive_qualification(values) == (False, "not_reinsurance_contract")

    values = _positive_values()
    values["relationship_term"] = "missing"
    values["business_covered_term"] = "unclear"
    values["overall_completeness"] = "unclear"
    assert derive_qualification(values) == (False, "insufficient_main_terms")


def test_repository_edge_case_seed_has_valid_evidence_and_derived_labels() -> None:
    repository = Path(__file__).parents[1]
    with (repository / "gold/edge-case-seed.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    cache: dict[str, int] = {}

    for row in rows:
        _validate_label_values(
            row,
            filename=row["download_filename"],
            download_dir=repository / "download",
            line_count_cache=cache,
        )
        qualifies, reason = derive_qualification(row)
        assert str(qualifies).lower() == row["qualifies"]
        assert reason == row["primary_rejection_reason"]
