# Independent review and adjudication workflow

The benchmark accepts only rows completed by two distinct reviewers and a distinct adjudicator. Historical GPT/Gemini fields and sampling strata remain in `gold/candidate-manifest.csv` for selection provenance; never give that unblinded file to reviewers.

## 1. Prepare independent sheets

First export the complete normalized documents. These are the authoritative line numbers used by the evidence columns and contain no weak labels or sampling strata:

```bash
uv run reinsurance-review export \
  --candidates gold/candidate-labeling.csv \
  --output-dir gold/review-packet
```

Create one sheet per reviewer from the blinded candidate file:

```bash
uv run reinsurance-review prepare \
  --candidates gold/candidate-labeling.csv \
  --reviewer-id reviewer-a \
  --output gold/review-a.csv

uv run reinsurance-review prepare \
  --candidates gold/candidate-labeling.csv \
  --reviewer-id reviewer-b \
  --output gold/review-b.csv
```

Reviewers work independently under `docs/LABELING_GUIDE.md`. Every evidence field must contain at least one normalized line reference such as `L42-L45: ...`. Reviewers must not see each other’s sheet, historical classifier outputs, selection strata, or holdout predictions.

## 2. Compare submissions

After both sheets are complete:

```bash
uv run reinsurance-review compare \
  --review-one gold/review-a.csv \
  --review-two gold/review-b.csv \
  --output gold/adjudication.csv
```

Comparison validates enum values, primary-rejection ordering, evidence line bounds against the normalized source document, reviewer identity, and identical document sets. Agreed values are copied into `final_*` columns; disagreements remain blank and are listed in `disagreement_fields`.

## 3. Adjudicate and finalize

A third person reviews every row, supplies a distinct `adjudicator` ID, fills any blank `final_*` values, and explains every disagreement in `adjudication_notes`. Then run:

```bash
uv run reinsurance-review finalize \
  --adjudication gold/adjudication.csv \
  --output gold/adjudicated.csv
```

Finalization revalidates all evidence, derives `qualifies` from the five gates, verifies the stable primary rejection reason, and emits the exact schema accepted by `reinsurance-benchmark`. It refuses missing identities, unresolved differences, out-of-range evidence, duplicate documents, or manually overridden qualification values.
