"""Local text normalization and criterion-balanced evidence selection."""

from __future__ import annotations

import html
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from bs4 import BeautifulSoup, NavigableString, Tag


ABSOLUTE_MAX_INPUT_CHARS = 32_000
DEFAULT_MAX_INPUT_CHARS = 24_000
DEFAULT_MAX_INPUT_TOKENS = 12_000
DEFAULT_MAX_OUTPUT_TOKENS = 2_500
DEFAULT_CONTEXT_SAFETY_TOKENS = 2_000
MAX_NORMALIZED_LINE_CHARS = 500


class EvidenceBudgetError(ValueError):
    pass


@dataclass(frozen=True)
class DocumentStats:
    source_bytes: int
    normalized_chars: int
    normalized_lines: int
    replacement_characters: int


@dataclass(frozen=True)
class NormalizedDocument:
    source_path: Path
    source_format: str
    text: str
    lines: tuple[str, ...]
    stats: DocumentStats


@dataclass(frozen=True)
class EvidencePack:
    text: str
    selected_line_numbers: frozenset[int]
    selected_ranges: tuple[tuple[int, int], ...]
    categories_found: tuple[str, ...]
    truncated: bool
    normalized_chars: int
    estimated_input_tokens: int


@dataclass(frozen=True)
class RequestLimits:
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    context_window_tokens: int = 32_768
    context_safety_tokens: int = DEFAULT_CONTEXT_SAFETY_TOKENS

    def __post_init__(self) -> None:
        if not 1_000 <= self.max_input_chars <= ABSOLUTE_MAX_INPUT_CHARS:
            raise ValueError(
                f"max_input_chars must be between 1000 and {ABSOLUTE_MAX_INPUT_CHARS}"
            )
        if self.max_input_tokens < 1 or self.max_output_tokens < 1:
            raise ValueError("token budgets must be positive")
        total = (
            self.max_input_tokens + self.max_output_tokens + self.context_safety_tokens
        )
        if total > self.context_window_tokens:
            raise ValueError("input/output budgets and safety margin exceed context window")


def conservative_token_estimate(text: str) -> int:
    """Use three characters per token when an exact model tokenizer is unavailable."""

    return math.ceil(len(text) / 3)


def normalize_file(path: Path) -> NormalizedDocument:
    raw = path.read_bytes()
    decoded = raw.decode("utf-8", errors="replace")
    suffix = path.suffix.lower()
    if suffix in {".htm", ".html"}:
        source_format = "html"
        extracted = _extract_html(decoded)
    elif suffix == ".txt":
        source_format = "text"
        extracted = decoded
    else:
        raise ValueError(f"unsupported source format: {path.suffix}")
    lines = tuple(_normalize_lines(extracted))
    text = "\n".join(lines)
    return NormalizedDocument(
        source_path=path,
        source_format=source_format,
        text=text,
        lines=lines,
        stats=DocumentStats(
            source_bytes=len(raw),
            normalized_chars=len(text),
            normalized_lines=len(lines),
            replacement_characters=decoded.count("\ufffd"),
        ),
    )


_HIDDEN_STYLE = re.compile(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", re.I)
_DROP_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
    "ix:header",
    "ix:hidden",
    "ix:references",
    "ix:resources",
}
_BLOCK_TAGS = {
    "address",
    "article",
    "blockquote",
    "caption",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "p",
    "pre",
    "section",
    "table",
    "tr",
    "ul",
    "ol",
}


def _extract_html(decoded: str) -> str:
    soup = BeautifulSoup(decoded, "lxml")
    for tag in list(soup.find_all(True)):
        # Descendants captured in this list can already be invalidated when a
        # hidden ancestor is decomposed earlier in the loop.
        if tag.name is None or tag.attrs is None:
            continue
        name = tag.name.lower()
        style = str(tag.attrs.get("style", ""))
        classes = {str(value).lower() for value in tag.attrs.get("class", [])}
        hidden = (
            name in _DROP_TAGS
            or tag.has_attr("hidden")
            or str(tag.attrs.get("aria-hidden", "")).lower() == "true"
            or bool(_HIDDEN_STYLE.search(style))
            or "hidden" in classes
        )
        if hidden:
            tag.decompose()
    for tag in list(soup.find_all(True)):
        if tag.name.lower() == "br":
            tag.replace_with(NavigableString("\n"))
        elif tag.name.lower() in {"td", "th"}:
            tag.append(NavigableString(" | "))
        elif tag.name.lower() in _BLOCK_TAGS:
            tag.append(NavigableString("\n"))
    return soup.get_text("")


def _normalize_lines(text: str) -> Iterable[str]:
    text = html.unescape(text).replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    for raw_line in text.splitlines():
        line = raw_line.strip(" |")
        if not line:
            continue
        chunks = textwrap.wrap(
            line,
            width=MAX_NORMALIZED_LINE_CHARS,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=True,
        )
        yield from chunks or [line]


_CATEGORY_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "document_relationship": tuple(
        re.compile(pattern, re.I)
        for pattern in (
            r"\breinsurance\b",
            r"\bretrocession\b",
            r"\bced(?:ent|ing company)\b",
            r"\breinsurer\b",
            r"\breinsured\b",
        )
    ),
    "document_kind": tuple(
        re.compile(pattern, re.I)
        for pattern in (
            r"\bamend(?:ment|ed)\b",
            r"\bendorsement\b",
            r"\bextension\b",
            r"\bcommutation\b",
            r"\bplacement slip\b",
            r"\bcover note\b",
            r"\bbinder\b",
        )
    ),
    "business": tuple(
        re.compile(pattern, re.I)
        for pattern in (
            r"business covered",
            r"class(?:es)? of business",
            r"polic(?:y|ies)",
            r"risks? (?:covered|insured|written)",
            r"subject business",
            r"account",
        )
    ),
    "term": tuple(
        re.compile(pattern, re.I)
        for pattern in (
            r"\bterm\b",
            r"\bperiod\b",
            r"\beffective\b",
            r"\binception\b",
            r"\bexpir(?:y|ation|es)\b",
            r"\bcommenc(?:e|es|ing)\b",
        )
    ),
    "economics": tuple(
        re.compile(pattern, re.I)
        for pattern in (
            r"\blimit\b",
            r"\bretention\b",
            r"\battachment\b",
            r"excess of",
            r"quota share",
            r"percentage share",
            r"reinsurer(?:'s)? liability",
        )
    ),
    "premium": tuple(
        re.compile(pattern, re.I)
        for pattern in (r"\bpremium\b", r"\bconsideration\b", r"deposit premium")
    ),
    "placement": tuple(
        re.compile(pattern, re.I)
        for pattern in (
            r"\bfacultative\b",
            r"\btreaty\b",
            r"\bautomatic(?:ally)?\b",
            r"\bobligatory\b",
            r"\bfacility\b",
            r"individual risk",
        )
    ),
    "government_pool": tuple(
        re.compile(pattern, re.I)
        for pattern in (
            r"\bstatut(?:e|ory)\b",
            r"\bgovernment\b",
            r"\bauthority\b",
            r"\bpool\b",
            r"\bfund\b",
            r"Florida Hurricane Catastrophe Fund",
        )
    ),
    "ending": tuple(
        re.compile(pattern, re.I)
        for pattern in (r"in witness whereof", r"\bsignature", r"\battachments?\b")
    ),
}


@dataclass(frozen=True)
class _Candidate:
    category: str
    start: int
    end: int


def build_evidence_pack(
    document: NormalizedDocument,
    metadata: dict[str, str],
    *,
    limits: RequestLimits | None = None,
    prompt_text: str = "",
    token_estimator: Callable[[str], int] = conservative_token_estimate,
) -> EvidencePack:
    limits = limits or RequestLimits()
    metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    all_numbers = set(range(1, len(document.lines) + 1))
    full_text = _render_pack(document, metadata_json, all_numbers, truncated=False)
    if _fits(full_text, prompt_text, limits, token_estimator):
        return EvidencePack(
            text=full_text,
            selected_line_numbers=frozenset(all_numbers),
            selected_ranges=_numbers_to_ranges(all_numbers),
            categories_found=tuple(_categories_present(document.lines)),
            truncated=False,
            normalized_chars=document.stats.normalized_chars,
            estimated_input_tokens=token_estimator(prompt_text + "\n" + full_text),
        )

    candidates = _build_candidates(document.lines)
    selected: set[int] = set()
    line_count = len(document.lines)
    mandatory = [
        _Candidate("preamble", 1, min(12, line_count)),
        _Candidate("ending", max(1, line_count - 14), line_count),
    ]
    for candidate in mandatory:
        _try_add_candidate(
            selected,
            candidate,
            document,
            metadata_json,
            prompt_text,
            limits,
            token_estimator,
        )

    grouped: dict[str, list[_Candidate]] = {key: [] for key in _CATEGORY_PATTERNS}
    for candidate in candidates:
        grouped[candidate.category].append(candidate)
    for group in grouped.values():
        del group[12:]
    while any(grouped.values()):
        made_progress = False
        for category in _CATEGORY_PATTERNS:
            if not grouped[category]:
                continue
            candidate = grouped[category].pop(0)
            made_progress |= _try_add_candidate(
                selected,
                candidate,
                document,
                metadata_json,
                prompt_text,
                limits,
                token_estimator,
            )
        if not made_progress and not any(grouped.values()):
            break

    if not selected:
        raise EvidenceBudgetError("metadata and prompt leave no room for exhibit evidence")
    rendered = _render_pack(document, metadata_json, selected, truncated=True)
    if not _fits(rendered, prompt_text, limits, token_estimator):
        raise EvidenceBudgetError("could not create an evidence pack within request limits")
    return EvidencePack(
        text=rendered,
        selected_line_numbers=frozenset(selected),
        selected_ranges=_numbers_to_ranges(selected),
        categories_found=tuple(
            category
            for category, group in _CATEGORY_PATTERNS.items()
            if any(
                any(pattern.search(document.lines[number - 1]) for pattern in group)
                for number in selected
            )
        ),
        truncated=True,
        normalized_chars=document.stats.normalized_chars,
        estimated_input_tokens=token_estimator(prompt_text + "\n" + rendered),
    )


def _fits(
    pack_text: str,
    prompt_text: str,
    limits: RequestLimits,
    token_estimator: Callable[[str], int],
) -> bool:
    if len(pack_text) > limits.max_input_chars:
        return False
    return token_estimator(prompt_text + "\n" + pack_text) <= limits.max_input_tokens


def _render_pack(
    document: NormalizedDocument,
    metadata_json: str,
    selected: set[int],
    *,
    truncated: bool,
) -> str:
    ranges = _numbers_to_ranges(selected)
    body: list[str] = []
    previous = 0
    for start, end in ranges:
        if previous and start > previous + 1:
            body.append(f"[OMITTED L{previous + 1:06d}-L{start - 1:06d}]")
        for number in range(start, end + 1):
            body.append(f"[L{number:06d}] {document.lines[number - 1]}")
        previous = end
    stats = {
        "source_format": document.source_format,
        "source_bytes": document.stats.source_bytes,
        "normalized_chars": document.stats.normalized_chars,
        "normalized_lines": document.stats.normalized_lines,
        "evidence_pack_truncated": truncated,
    }
    return (
        "<document-metadata>\n"
        + metadata_json
        + "\n</document-metadata>\n<document-statistics>\n"
        + json.dumps(stats, sort_keys=True)
        + "\n</document-statistics>\n<document-text>\n"
        + "\n".join(body)
        + "\n</document-text>"
    )


def _categories_present(lines: tuple[str, ...]) -> Iterable[str]:
    for category, patterns in _CATEGORY_PATTERNS.items():
        if any(any(pattern.search(line) for pattern in patterns) for line in lines):
            yield category


def _build_candidates(lines: tuple[str, ...]) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for index, line in enumerate(lines, start=1):
        for category, patterns in _CATEGORY_PATTERNS.items():
            if any(pattern.search(line) for pattern in patterns):
                radius = 4 if category in {"business", "economics", "premium"} else 3
                # Seed every criterion with a compact local excerpt before
                # spending remaining room on wider context windows.
                candidates.append(
                    _Candidate(
                        category,
                        max(1, index - 1),
                        min(len(lines), index + 1),
                    )
                )
                candidates.append(
                    _Candidate(
                        category,
                        max(1, index - radius),
                        min(len(lines), index + radius),
                    )
                )
        if "table of contents" in line.lower() or line.strip().lower() == "contents":
            candidates.append(
                _Candidate("document_kind", index, min(len(lines), index + 25))
            )
    return _deduplicate_candidates(candidates)


def _deduplicate_candidates(candidates: list[_Candidate]) -> list[_Candidate]:
    seen: set[tuple[str, int, int]] = set()
    result: list[_Candidate] = []
    for candidate in candidates:
        key = (candidate.category, candidate.start, candidate.end)
        if key not in seen:
            seen.add(key)
            result.append(candidate)
    return result


def _try_add_candidate(
    selected: set[int],
    candidate: _Candidate,
    document: NormalizedDocument,
    metadata_json: str,
    prompt_text: str,
    limits: RequestLimits,
    token_estimator: Callable[[str], int],
) -> bool:
    additions = set(range(candidate.start, candidate.end + 1)) - selected
    if not additions:
        return False
    tentative = selected | additions
    rendered = _render_pack(document, metadata_json, tentative, truncated=True)
    if not _fits(rendered, prompt_text, limits, token_estimator):
        return False
    selected.update(additions)
    return True


def _numbers_to_ranges(numbers: set[int]) -> tuple[tuple[int, int], ...]:
    if not numbers:
        return ()
    ordered = sorted(numbers)
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for number in ordered[1:]:
        if number == previous + 1:
            previous = number
            continue
        ranges.append((start, previous))
        start = previous = number
    ranges.append((start, previous))
    return tuple(ranges)
