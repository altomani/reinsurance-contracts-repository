# Local corpus preparation report

Date: 2026-08-29  
Prompt: `classifier-v1`  
API requests and cost: 0

## Scope and outcome

The classifier completed a no-API metadata traversal, text normalization, stable line numbering, and evidence-pack build for the current corpus. The index contains 6,763 rows but only 6,761 distinct `downloadFilename` values. Two duplicate rows point to the same downloaded exhibits; the metadata loader now deduplicates those filenames before classification.

| Latest status (unique filename) | Count |
| --- | ---: |
| Prepared supported non-PDF exhibit | 6,678 |
| Missing downloaded file | 64 |
| Skipped PDF | 19 |
| Remaining extraction error | 0 |
| Total | 6,761 |

The downloaded directory contains 5,289 `.htm`, one `.html`, 1,388 `.txt`, and 19 `.pdf` files. The 6,678 prepared non-PDF exhibits therefore cover every downloaded supported file exactly once.

## Request-size results

The configured evidence-pack ceiling was 24,000 characters (absolute implementation ceiling: 32,000). The configured input-token ceiling was 12,000, estimated conservatively at three characters per token and including the versioned prompt.

| Measure | Min | P50 | P90 | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normalized document characters | 275 | 85,638 | 597,615 | 777,591 | 1,131,832 | 4,731,135 |
| Evidence-pack characters | 1,193 | 22,572 | 23,985 | 23,992 | 23,999 | 24,000 |
| Estimated request input tokens | 1,150 | 8,276 | 8,747 | 8,749 | 8,752 | 8,752 |

All 6,678 packs respected both hard limits. No raw document was submitted. Full normalized text fit for 1,147 exhibits; 5,531 required criterion-balanced selection. Selected-line counts were 195 at P50, 325 at P90, 353 at P95, and 1,000 maximum.

## Evidence-category coverage

These counts mean that at least one selected line matched the category’s high-recall lexical detector. They demonstrate selection behavior, not that the legal criterion is satisfied.

| Evidence category | Packs containing a selected match |
| --- | ---: |
| Reinsurance relationship | 6,677 |
| Document-kind indicators | 6,393 |
| Covered business | 6,260 |
| Term/period | 6,605 |
| Risk economics | 5,757 |
| Premium/consideration | 6,054 |
| Treaty/facultative placement | 4,886 |
| Government fund/pool | 5,965 |
| Ending/signature/attachment | 6,034 |

Decisive-term survival must be measured against the adjudicated gold evidence lines. The benchmark utility computes that coverage by normalized-length bin once those labels exist.

## Defects found and resolved

The first traversal found a BeautifulSoup edge case in which decomposing a hidden ancestor invalidated descendant tag objects already queued for inspection. A guard for invalidated tags was added and regression-tested. All 117 affected rows then succeeded on resume; prepared/missing/PDF rows were not repeated.

The traversal also found two duplicate metadata rows. The loader now emits each nonempty `downloadFilename` once, avoiding duplicate paid calls.

## Next quality gate

`gold/candidate-manifest.csv` contains 180 deterministic candidates: 119 development and 61 holdout, stratified across historical disagreements, amendments/endorsements, placement slips, life/health/annuity, facultative edges, government schemes, clear treaty candidates, unrelated agreements, years, formats, and size bins. Historical outputs are selection provenance only. Reviewers should receive `gold/candidate-labeling.csv`, which omits historical labels and selection strata.

The candidate set still requires independent review and adjudication under `docs/LABELING_GUIDE.md`. No paid benchmark, prompt freeze, pilot, forecast, or full-corpus run is authorized by this preparation result alone.

The independent-review exporter was also exercised on the full blinded candidate set. It produced all 180 complete normalized, line-numbered documents plus a leakage-free manifest: 29,479,359 normalized characters and approximately 34 MB on disk. The packet itself is generated locally and is not committed to the repository.
