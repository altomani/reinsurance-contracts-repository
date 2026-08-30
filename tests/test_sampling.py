from __future__ import annotations

import csv
from pathlib import Path

from reinsurance_classifier.sampling import (
    build_candidate_manifest,
    manifest_summary,
    write_blinded_label_sheet,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_candidate_sampling_is_reproducible_and_does_not_create_gold_labels(
    tmp_path: Path,
) -> None:
    index = tmp_path / "index"
    download = tmp_path / "download"
    gpt = tmp_path / "gpt"
    gemini = tmp_path / "gemini"
    for directory in (index, download, gpt, gemini):
        directory.mkdir()
    metadata = []
    gpt_rows = []
    gemini_rows = []
    descriptions = [
        "PROPERTY CATASTROPHE REINSURANCE CONTRACT",
        "ENDORSEMENT TO REINSURANCE CONTRACT",
        "FLORIDA HURRICANE CATASTROPHE FUND REIMBURSEMENT CONTRACT",
        "STOCK PURCHASE AGREEMENT",
    ]
    for number in range(16):
        filename = f"2024-{number}.txt"
        metadata.append({"downloadFilename": filename, "description": descriptions[number % 4]})
        gpt_rows.append(
            {
                "downloadFilename": filename,
                "reinsurance": "Yes",
                "contractType": "Non-Life",
                "obligatoryType": "Treaty",
                "classOfBusiness": "Property",
            }
        )
        gemini_rows.append(
            {
                "downloadFilename": filename,
                "reinsurance": "No" if number == 0 else "Yes",
                "contractType": "Non-Life",
                "obligatoryType": "Treaty",
                "classOfBusiness": "Property",
            }
        )
        (download / filename).write_text("document", encoding="utf-8")
    _write_csv(index / "index-2024.csv", metadata)
    _write_csv(gpt / "index-2024.csv", gpt_rows)
    _write_csv(gemini / "index-2024.csv", gemini_rows)

    kwargs = {
        "index_dir": index,
        "download_dir": download,
        "gpt_dir": gpt,
        "gemini_dir": gemini,
        "target": 12,
        "seed": 7,
    }
    first = build_candidate_manifest(**kwargs)
    second = build_candidate_manifest(**kwargs)

    assert first == second
    assert len(first) == 12
    assert all("qualifies" not in row for row in first)
    assert "historical_disagreement" in manifest_summary(first)["strata"]

    blinded = tmp_path / "blinded.csv"
    write_blinded_label_sheet(first, blinded)
    with blinded.open(newline="", encoding="utf-8") as handle:
        blinded_rows = list(csv.DictReader(handle))
    assert len(blinded_rows) == 12
    assert "gpt_reinsurance" not in blinded_rows[0]
    assert all(not row["selection_stratum"] for row in blinded_rows)
    assert {row["split"] for row in blinded_rows} == {"development", "holdout"}
