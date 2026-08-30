from __future__ import annotations

import re
from pathlib import Path

import pytest

from reinsurance_classifier.extraction import (
    EvidenceBudgetError,
    RequestLimits,
    build_evidence_pack,
    conservative_token_estimate,
    normalize_file,
)


def test_html_normalization_removes_noise_and_preserves_structure(tmp_path: Path) -> None:
    path = tmp_path / "sample.htm"
    path.write_text(
        """
        <html><head><style>.x {display:none}</style><script>BAD SCRIPT</script></head>
        <body><h1>REINSURANCE AGREEMENT</h1>
        <div hidden>HIDDEN ATTRIBUTE <span>HIDDEN CHILD</span></div>
        <p style="display:none">HIDDEN STYLE</p>
        <ix:hidden>INLINE XBRL NOISE</ix:hidden>
        <p>The Cedent cedes risks.<br>Effective January 1.</p>
        <table><tr><td>Limit</td><td>$5,000,000</td></tr></table>
        </body></html>
        """,
        encoding="utf-8",
    )

    document = normalize_file(path)

    assert "BAD SCRIPT" not in document.text
    assert "HIDDEN ATTRIBUTE" not in document.text
    assert "HIDDEN CHILD" not in document.text
    assert "HIDDEN STYLE" not in document.text
    assert "INLINE XBRL NOISE" not in document.text
    assert "REINSURANCE AGREEMENT" in document.lines
    assert any("Effective January 1" in line for line in document.lines)
    assert any("Limit" in line and "$5,000,000" in line for line in document.lines)


def test_small_document_is_submitted_in_full_with_stable_lines(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("Treaty\nPremium is redacted.\nIn witness whereof", encoding="utf-8")
    document = normalize_file(path)

    pack = build_evidence_pack(document, {"description": "sample"})

    assert pack.truncated is False
    assert pack.selected_line_numbers == frozenset({1, 2, 3})
    assert "[L000001] Treaty" in pack.text
    assert "[L000003] In witness whereof" in pack.text


def test_long_document_balances_categories_and_deduplicates_lines(tmp_path: Path) -> None:
    path = tmp_path / "long.txt"
    lines = [f"Boilerplate line {number} " + "x" * 80 for number in range(1, 220)]
    replacements = {
        5: "REINSURANCE AGREEMENT between the Ceding Company and the Reinsurer",
        42: "BUSINESS COVERED: all property policies in the account",
        76: "TERM: effective January 1 and expires December 31",
        109: "LIMIT AND RETENTION: $5m excess of $2m",
        143: "PREMIUM: the deposit premium is [REDACTED]",
        176: "AUTOMATIC TREATY: all qualifying risks are automatically ceded",
        198: "This treaty excludes the Florida Hurricane Catastrophe Fund.",
        217: "IN WITNESS WHEREOF the parties sign below",
    }
    for line_number, value in replacements.items():
        lines[line_number - 1] = value
    path.write_text("\n".join(lines), encoding="utf-8")
    document = normalize_file(path)
    limits = RequestLimits(max_input_chars=7_500, max_input_tokens=3_500)

    pack = build_evidence_pack(document, {"description": "long"}, limits=limits)

    assert pack.truncated is True
    assert len(pack.text) <= limits.max_input_chars
    assert pack.estimated_input_tokens <= limits.max_input_tokens
    for required in {"business", "term", "economics", "premium", "placement"}:
        assert required in pack.categories_found
    numbered = re.findall(r"^\[L(\d{6})\]", pack.text, re.MULTILINE)
    assert len(numbered) == len(set(numbered))
    assert "Florida Hurricane Catastrophe Fund" in pack.text
    assert "IN WITNESS WHEREOF" in pack.text


def test_token_cap_is_independent_of_character_cap(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("\n".join(f"premium line {i}" for i in range(100)), encoding="utf-8")
    document = normalize_file(path)
    limits = RequestLimits(max_input_chars=10_000, max_input_tokens=500)

    pack = build_evidence_pack(document, {}, limits=limits)

    assert conservative_token_estimate(pack.text) <= limits.max_input_tokens


def test_request_limit_validation_and_too_large_prompt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_input_chars"):
        RequestLimits(max_input_chars=32_001)

    path = tmp_path / "sample.txt"
    path.write_text("reinsurance agreement", encoding="utf-8")
    document = normalize_file(path)
    with pytest.raises(EvidenceBudgetError):
        build_evidence_pack(
            document,
            {},
            limits=RequestLimits(max_input_chars=1_000, max_input_tokens=10),
            prompt_text="z" * 1_000,
        )
