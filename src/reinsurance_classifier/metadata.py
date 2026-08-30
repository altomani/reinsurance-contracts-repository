"""Load yearly SEC metadata and resolve downloaded exhibits."""

from __future__ import annotations

import csv
import re
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Iterator

from pydantic import Field

from .models import StrictModel


class SourceStatus(StrEnum):
    SUPPORTED = "supported"
    MISSING_FILE = "missing_file"
    SKIPPED_PDF = "skipped_pdf"
    UNSUPPORTED_FILE = "unsupported_file"


class MetadataRecord(StrictModel):
    record_id: str
    year: int
    row_number: int = Field(ge=2)
    download_filename: str
    source_path: Path | None
    source_status: SourceStatus
    metadata: dict[str, str]


def _index_year(path: Path) -> int | None:
    match = re.fullmatch(r"index-(\d{4})\.csv", path.name)
    return int(match.group(1)) if match else None


def discover_index_files(index_dir: Path, years: set[int] | None = None) -> list[Path]:
    found: list[tuple[int, Path]] = []
    for path in index_dir.glob("index-*.csv"):
        year = _index_year(path)
        if year is not None and (years is None or year in years):
            found.append((year, path))
    return [path for _, path in sorted(found)]


def load_metadata_records(
    index_dir: Path,
    download_dir: Path,
    *,
    years: Iterable[int] | None = None,
    filenames: Iterable[str] | None = None,
    limit: int | None = None,
) -> Iterator[MetadataRecord]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    year_filter = set(years) if years is not None else None
    file_filter = set(filenames) if filenames is not None else None
    emitted = 0
    seen_filenames: set[str] = set()
    for index_path in discover_index_files(index_dir, year_filter):
        year = _index_year(index_path)
        assert year is not None
        with index_path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "downloadFilename" not in reader.fieldnames:
                raise ValueError(f"missing downloadFilename column: {index_path}")
            for row_number, raw_row in enumerate(reader, start=2):
                row = {key: value or "" for key, value in raw_row.items() if key is not None}
                filename = row.get("downloadFilename", "").strip()
                if file_filter is not None and filename not in file_filter:
                    continue
                if filename and filename in seen_filenames:
                    continue
                if filename:
                    seen_filenames.add(filename)
                source_path, status = _resolve_source(download_dir, filename)
                yield MetadataRecord(
                    record_id=f"{year}:{row_number}:{filename or '[missing-name]'}",
                    year=year,
                    row_number=row_number,
                    download_filename=filename,
                    source_path=source_path,
                    source_status=status,
                    metadata=row,
                )
                emitted += 1
                if limit is not None and emitted >= limit:
                    return


def _resolve_source(download_dir: Path, filename: str) -> tuple[Path | None, SourceStatus]:
    if not filename or Path(filename).name != filename:
        return None, SourceStatus.MISSING_FILE
    path = download_dir / filename
    if not path.is_file():
        return path, SourceStatus.MISSING_FILE
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return path, SourceStatus.SKIPPED_PDF
    if suffix not in {".txt", ".htm", ".html"}:
        return path, SourceStatus.UNSUPPORTED_FILE
    return path, SourceStatus.SUPPORTED
