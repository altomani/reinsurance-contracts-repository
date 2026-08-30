"""Build auditable autonomous silver labels from model and prompt concordance."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .aggregation import CriterionValue, decision_criteria, has_rule_contradiction
from .output import read_latest_records


CRITERIA = (
    "document_contract",
    "main_terms",
    "non_life_business",
    "treaty_placement",
    "private_market",
)


@dataclass(frozen=True)
class Vote:
    prompt: str
    model: str
    values: tuple[CriterionValue, ...]
    valid: bool


def build_report(
    *, manifest_path: Path, audit_paths: list[Path], output_csv: Path
) -> dict:
    manifest = _read_manifest(manifest_path)
    votes: dict[str, list[Vote]] = defaultdict(list)
    hashes: dict[str, set[str]] = defaultdict(set)
    calls = invalid_calls = quote_invalid_calls = 0
    total_cost = 0.0
    prompt_names: set[str] = set()
    model_names: set[str] = set()
    seen_rating: set[tuple[str, str, str]] = set()

    for audit_path in audit_paths:
        for record in read_latest_records(audit_path).values():
            filename = record.download_filename
            if filename not in manifest:
                continue
            prompt_names.add(record.prompt_version)
            if record.source_sha256:
                hashes[filename].add(record.source_sha256)
            for call in record.model_calls:
                key = (filename, record.prompt_version, call.model_id)
                if key in seen_rating:
                    raise ValueError(f"duplicate rating for {key}")
                seen_rating.add(key)
                criteria = decision_criteria(call.decision)
                valid = (
                    (call.evidence_lines_valid or call.evidence_quotes_valid)
                    and not has_rule_contradiction(call.decision)
                )
                votes[filename].append(
                    Vote(
                        prompt=record.prompt_version,
                        model=call.model_id,
                        values=criteria.values(),
                        valid=valid,
                    )
                )
                calls += 1
                invalid_calls += int(not valid)
                quote_invalid_calls += int(not call.evidence_quotes_valid)
                total_cost += call.cost_usd or 0.0
                model_names.add(call.model_id)

    rows: list[dict[str, str]] = []
    resolved = qualified = rejected = unresolved = 0
    exact_vectors = 0
    prompt_stable_pairs = prompt_pairs = 0
    criterion_resolved = Counter()
    criterion_agreement = Counter()

    for filename, meta in manifest.items():
        document_votes = votes.get(filename, [])
        valid_votes = [vote for vote in document_votes if vote.valid]
        final_values: list[CriterionValue] = []
        vote_fields: dict[str, str] = {}
        for index, criterion in enumerate(CRITERIA):
            value, count = _quorum_value(valid_votes, index, quorum=4)
            final_values.append(value)
            criterion_resolved[criterion] += int(value != CriterionValue.UNCLEAR)
            criterion_agreement[criterion] += count
            vote_fields[f"{criterion}_votes"] = _vote_summary(valid_votes, index)

        status = _outcome(final_values)
        resolved += int(status != "manual_review")
        qualified += int(status == "qualified")
        rejected += int(status == "rejected")
        unresolved += int(status == "manual_review")
        vectors = [vote.values for vote in valid_votes]
        exact_vectors += int(len(vectors) >= 4 and len(set(vectors)) == 1)

        by_model: dict[str, dict[str, tuple[CriterionValue, ...]]] = defaultdict(dict)
        for vote in valid_votes:
            by_model[vote.model][vote.prompt] = vote.values
        for per_prompt in by_model.values():
            if len(per_prompt) >= 2:
                prompt_pairs += 1
                prompt_stable_pairs += int(len(set(per_prompt.values())) == 1)

        rows.append(
            {
                "document_id": meta.get("document_id", ""),
                "year": meta.get("year", ""),
                "download_filename": filename,
                "split": meta.get("split", ""),
                "selection_stratum": meta.get("selection_stratum", ""),
                "label_provenance": "autonomous_silver_v1",
                "valid_votes": str(len(valid_votes)),
                "invalid_votes": str(len(document_votes) - len(valid_votes)),
                "source_hash_consistent": str(len(hashes.get(filename, set())) <= 1).lower(),
                **{
                    criterion: value.value
                    for criterion, value in zip(CRITERIA, final_values, strict=True)
                },
                "status": status,
                "qualifies": str(status == "qualified").lower(),
                **vote_fields,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    return {
        "label_provenance": "autonomous_silver_v1",
        "limitations": (
            "Silver labels are deterministic supermajority concordance, not human gold. "
            "Quality estimates use model-excluded consensus."
        ),
        "documents": len(manifest),
        "prompts": sorted(prompt_names),
        "models": sorted(model_names),
        "calls": calls,
        "hard_invalid_evidence_or_rule_calls": invalid_calls,
        "hard_validation_rate": (calls - invalid_calls) / calls if calls else 0.0,
        "quote_grounding_rate": (
            (calls - quote_invalid_calls) / calls if calls else 0.0
        ),
        "source_hash_conflicts": sum(len(value) > 1 for value in hashes.values()),
        "total_cost_usd": total_cost,
        "labels": {
            "resolved": resolved,
            "qualified": qualified,
            "rejected": rejected,
            "manual_review": unresolved,
            "exact_six_vote_vectors": exact_vectors,
        },
        "criterion_resolution_rate": {
            name: criterion_resolved[name] / len(manifest) for name in CRITERIA
        },
        "prompt_vector_stability": (
            prompt_stable_pairs / prompt_pairs if prompt_pairs else 0.0
        ),
        "model_excluded_benchmark": _model_excluded_benchmark(votes, manifest),
        "output_csv": str(output_csv),
    }


def _quorum_value(
    votes: list[Vote], criterion_index: int, *, quorum: int, min_models: int = 3
) -> tuple[CriterionValue, int]:
    candidates = Counter(vote.values[criterion_index] for vote in votes)
    for value, count in candidates.most_common():
        supporters = [vote for vote in votes if vote.values[criterion_index] == value]
        if (
            count >= quorum
            and len({vote.model for vote in supporters}) >= min_models
            and len({vote.prompt for vote in supporters}) >= 2
        ):
            return value, count
    return CriterionValue.UNCLEAR, candidates.most_common(1)[0][1] if candidates else 0


def _vote_summary(votes: list[Vote], criterion_index: int) -> str:
    counts = Counter(vote.values[criterion_index].value for vote in votes)
    return ";".join(f"{value}:{counts.get(value, 0)}" for value in ("pass", "fail", "unclear"))


def _outcome(values: list[CriterionValue]) -> str:
    if all(value == CriterionValue.PASS for value in values):
        return "qualified"
    if CriterionValue.FAIL in values:
        return "rejected"
    return "manual_review"


def _model_excluded_benchmark(
    votes: dict[str, list[Vote]], manifest: dict[str, dict[str, str]]
) -> dict:
    pairs: dict[tuple[str, str, str], list[tuple[bool, bool]]] = defaultdict(list)
    for filename, document_votes in votes.items():
        split = manifest[filename].get("split", "")
        valid_votes = [vote for vote in document_votes if vote.valid]
        for target in valid_votes:
            others = [vote for vote in valid_votes if vote.model != target.model]
            truth_values = [
                _quorum_value(others, index, quorum=3, min_models=2)[0]
                for index in range(len(CRITERIA))
            ]
            truth_status = _outcome(truth_values)
            if truth_status == "manual_review":
                continue
            predicted = all(value == CriterionValue.PASS for value in target.values)
            expected = truth_status == "qualified"
            pairs[(split, target.prompt, target.model)].append((expected, predicted))
    return {
        f"{split}|{prompt}|{model}": _binary_metrics(values)
        for (split, prompt, model), values in sorted(pairs.items())
    }


def _binary_metrics(pairs: list[tuple[bool, bool]]) -> dict:
    tp = sum(expected and predicted for expected, predicted in pairs)
    fp = sum(not expected and predicted for expected, predicted in pairs)
    tn = sum(not expected and not predicted for expected, predicted in pairs)
    fn = sum(expected and not predicted for expected, predicted in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": len(pairs),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": precision,
        "precision_95_ci": _wilson(tp, tp + fp),
        "recall": recall,
        "recall_95_ci": _wilson(tp, tp + fn),
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if not total:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _read_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = {row.get("download_filename", "").strip(): row for row in rows}
    result.pop("", None)
    if not result:
        raise ValueError("manifest contains no download_filename values")
    if len(result) != len(rows):
        raise ValueError("manifest contains duplicate or empty download_filename values")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_report(
            manifest_path=args.manifest,
            audit_paths=args.audit,
            output_csv=args.output_csv,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output_report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
