# Reinsurance Contracts Repository

A corpus of publicly available reinsurance-related exhibits retrieved from SEC filings, together with metadata and auditable classification outputs. The dataset is also published on [Hugging Face](https://huggingface.co/datasets/andreaaltomani/reinsurance-contracts-classification).

## Repository layout

| Path | Contents |
| --- | --- |
| `download/` | Original HTML, TXT, and PDF exhibits from 2001–2024 |
| `index-download/` | Yearly SEC metadata and local download filenames |
| `src/reinsurance_classifier/` | Maintained evidence-based classifier and command-line tools |
| `prompts/` | Versioned production and falsification prompts |
| `results/final/` | Canonical latest-state CSV and append-only audit JSONL |
| `reports/` | Validation, integrity, preparation, and cost reports |
| `gold/` | Candidate manifests, labeling sheets, and autonomous-silver labels |
| `docs/` | Classifier plan, labeling policy, and review protocols |
| `archive/classifiers/` | Superseded classifier scripts and historical per-model results |
| `scripts/` | The SEC downloader and current diagnostic utilities |

Generated preparation and validation runs may be kept locally under `results/preparation/` and `results/validation/`; they are ignored by Git because the tracked reports contain their durable summaries.

## Current classifier

The maintained classifier converts each supported exhibit into a criterion-balanced evidence pack with stable line numbers. A document qualifies only when all five gates pass:

1. it is a complete or nearly complete reinsurance contract;
2. the main commercial terms are present;
3. the covered business is non-life;
4. the placement is treaty or automatic rather than facultative; and
5. it is not a statutory government scheme.

Python enforces this conjunction after structured model review. A model cannot override it. Provider evidence, token use, cost, retries, and append-only corrections remain available in the audit output. Unsupported PDFs and missing downloads are recorded without failing the run.

The implementation and governance material live in:

- `docs/CLASSIFIER_PLAN.md` — quality, pilot, budget, and rollout gates;
- `docs/LABELING_GUIDE.md` — adjudication policy;
- `docs/REVIEW_WORKFLOW.md` — independent review and adjudication;
- `docs/AUTONOMOUS_VALIDATION.md` — conservative autonomous-silver fallback;
- `prompts/` — frozen prompt versions;
- `reports/` — machine-readable and narrative run evidence.

## Final classification run

The 2026-08-30 run processed all 6,761 unique metadata records. Its latest state contains 651 automatic qualifications, 342 rejections, 5,685 manual-review cases, 64 missing downloads, and 19 skipped PDFs, with no remaining processing errors.

Canonical deliverables:

- `results/final/classification-latest.csv` — one filter-friendly latest row per record;
- `results/final/classification-audit.jsonl` — append-only decisions, evidence, provider metadata, cost, retries, and corrections;
- `reports/final-classification-report.md` — final counts, validation, budget reconciliation, and limitations;
- `reports/final-integrity.json` — machine-readable integrity and budget checks.

The reported accuracy evidence is conservative autonomous-silver concordance, not independent human gold. Unresolved cases remain `manual_review`; they are never forced into positive or negative labels.

## Setup and verification

Python 3.13 and [uv](https://docs.astral.sh/uv/) are expected:

```bash
uv sync
uv run pytest
```

Prepare evidence packs locally without making API calls:

```bash
uv run reinsurance-classifier --dry-run --year 2004 --limit 25
```

A bounded paid run requires `OPENROUTER_API_KEY` in `.env` or the environment:

```bash
uv run reinsurance-classifier --year 2004 --limit 100 --budget-usd 5
```

The CLI also supports repeated year and file selectors, ordered model routes, configurable concurrency, resume, independent character and token limits, and a hard dollar ceiling. An unbounded run additionally requires `--allow-full-corpus` and a compatible permitted gate report; see `docs/CLASSIFIER_PLAN.md` before spending API credits.

Evaluate audit records against adjudicated labels:

```bash
uv run reinsurance-benchmark \
  --gold gold/adjudicated.csv \
  --audit results/classification-audit.jsonl \
  --split holdout
```

Build the deterministic candidate sample or prepare independent reviewer sheets:

```bash
uv run reinsurance-sample --target 180

uv run reinsurance-review prepare \
  --candidates gold/candidate-labeling.csv \
  --reviewer-id reviewer-a \
  --output gold/review-a.csv
```

See `docs/REVIEW_WORKFLOW.md` for comparison and adjudication. Historical model results used for disagreement sampling remain available under `archive/classifiers/results/`.

## Historical archive

Superseded one-off classifier and join scripts, along with their yearly model outputs, are retained under `archive/classifiers/` for provenance. They are not part of the maintained environment or current workflow. See `archive/classifiers/README.md` for the archive map and legacy dependency notes.
