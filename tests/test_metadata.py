from __future__ import annotations

import csv
from pathlib import Path

from reinsurance_classifier.metadata import SourceStatus, load_metadata_records


def test_metadata_join_records_missing_pdf_and_supported_files(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    download_dir = tmp_path / "download"
    index_dir.mkdir()
    download_dir.mkdir()
    (download_dir / "contract.htm").write_text("agreement", encoding="utf-8")
    (download_dir / "scan.pdf").write_bytes(b"%PDF")
    with (index_dir / "index-2024.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["downloadFilename", "description"])
        writer.writeheader()
        writer.writerows(
            [
                {"downloadFilename": "contract.htm", "description": "supported"},
                {"downloadFilename": "scan.pdf", "description": "pdf"},
                {"downloadFilename": "missing.txt", "description": "missing"},
            ]
        )

    records = list(load_metadata_records(index_dir, download_dir))

    assert [record.source_status for record in records] == [
        SourceStatus.SUPPORTED,
        SourceStatus.SKIPPED_PDF,
        SourceStatus.MISSING_FILE,
    ]
    assert records[0].year == 2024
    assert records[0].row_number == 2


def test_year_file_and_limit_filters_are_applied_before_emission(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    download_dir = tmp_path / "download"
    index_dir.mkdir()
    download_dir.mkdir()
    for year in (2023, 2024):
        with (index_dir / f"index-{year}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["downloadFilename"])
            writer.writeheader()
            writer.writerow({"downloadFilename": f"{year}.txt"})
        (download_dir / f"{year}.txt").write_text("text", encoding="utf-8")

    records = list(
        load_metadata_records(
            index_dir,
            download_dir,
            years=[2024],
            filenames=["2024.txt"],
            limit=1,
        )
    )

    assert len(records) == 1
    assert records[0].year == 2024


def test_duplicate_download_filenames_are_classified_once(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    download_dir = tmp_path / "download"
    index_dir.mkdir()
    download_dir.mkdir()
    (download_dir / "same.txt").write_text("text", encoding="utf-8")
    with (index_dir / "index-2024.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["downloadFilename", "type"])
        writer.writeheader()
        writer.writerow({"downloadFilename": "same.txt", "type": "EX-10.1"})
        writer.writerow({"downloadFilename": "same.txt", "type": "EX-10.2"})

    records = list(load_metadata_records(index_dir, download_dir))

    assert len(records) == 1
    assert records[0].download_filename == "same.txt"
