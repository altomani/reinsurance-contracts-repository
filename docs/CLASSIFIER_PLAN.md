# Reinsurance contract classifier: implementation plan

## Objective

Classify each non-PDF EDGAR exhibit as qualifying only when all five conditions are met:

1. The document itself is a reinsurance or retrocession contract, rather than an amendment, endorsement, extension, or merely related document.
2. It is complete or nearly complete and contains the main commercial terms. A redacted premium is acceptable.
3. The covered business is non-life, including health written on a non-life basis, but excluding life-like health, health plans, life, annuity, and pension business.
4. The placement is treaty/automatic rather than facultative.
5. It is not a statutory government reinsurance scheme.

The final `qualifies` value will be computed in Python as the conjunction of the five criterion results. The model will not be allowed to override that invariant.

Missing or redacted counterparty names are acceptable and must not make an otherwise complete contract fail. The classifier needs evidence of a reinsurance risk-transfer relationship and the relevant roles, but it does not require the cedent or reinsurer's legal identity to be disclosed.

## Repository observations

- `download/` contains 6,697 files: 5,289 `.htm`, 1 `.html`, 1,388 `.txt`, and 19 `.pdf` files. PDFs will initially receive a `skipped_pdf` status.
- `index-download/` contains 6,761 metadata rows; 64 rows do not currently have a corresponding downloaded file.
- HTML file sizes are highly skewed: median 270 KB, 90th percentile 1.77 MB, and maximum 33.8 MB. Sending raw files is therefore neither economical nor robust.
- Some raw files exceed even the models' large advertised context windows. No raw document may be sent directly to a model; every supported document must first pass through normalization, section selection, and a hard request-size guard.
- The older GPT and Gemini outputs are useful as weak labels for stratified sampling, but not as ground truth. On rows where both produced values, they agree on `reinsurance` only 76.5% of the time and on `classOfBusiness` only 47.2% of the time. Their questions also do not test completeness or the government-scheme exclusion.
- Reviewed edge cases include a 2 KB facultative endorsement, a life coinsurance agreement, a complete private property-catastrophe treaty, a seven-page placement slip, a Florida Hurricane Catastrophe Fund contract, and unrelated credit, stock-purchase, and settlement agreements.

## Model and provider configuration

Use the current OpenRouter model identifiers and only their first-party endpoints:

| Model | OpenRouter ID | Required provider slug |
| --- | --- | --- |
| Qwen 3.8 Flash | `qwen/qwen3.8-flash` | `alibaba` |
| DeepSeek V4 Flash Vision Exp | `deepseek/deepseek-v4-flash-vision-exp` | `deepseek` |
| GLM 5.3 Flash | `z-ai/glm-5.3-flash` | `z-ai` |

Every request will set `only: [provider_slug]` and `allow_fallbacks: false`. The smoke test confirmed the downstream providers as Alibaba, DeepSeek, and Z.AI respectively.

These connectivity checks are complete and should not be repeated by the next instance unless the model configuration changes or a real classification request fails with a routing/authentication error. Re-running them adds cost without new evidence.

Pydantic AI prompted JSON output will be used for the shared path. This was the only mode that worked unchanged across all three routes: Alibaba rejects a required output tool while reasoning is enabled, and Z.AI requires reasoning while also requiring tool choice to remain automatic. Pydantic will still validate the JSON against the output model and retry malformed responses.

## Classification schema

The structured response should preserve the individual decisions and evidence needed to audit them:

- `document_kind`: `complete_contract`, `nearly_complete_contract`, `amendment_or_endorsement`, `placement_slip_or_summary`, `related_other`, `unrelated`, or `unclear`.
- `is_reinsurance_contract`: yes/no/unclear.
- `main_terms`:
  - reinsurance relationship and party roles; actual party names may be named, redacted, or absent;
  - business covered;
  - term/period;
  - limit, retention, attachment, share, or other applicable risk transfer economics;
  - premium/consideration, with `redacted` accepted;
  - overall completeness.
- `business_basis`: `non_life`, `life_like`, `mixed`, or `unclear`.
- `placement_basis`: `treaty`, `facultative`, `mixed`, or `unclear`.
- `counterparty_disclosure`: `named`, `partly_redacted`, `fully_redacted`, `missing`, or `unclear`. This is informational and does not affect `qualifies`.
- `government_basis`: `private_market`, `statutory_government_scheme`, or `unclear`.
- `pool_or_scheme`, containing:
  - `involvement`: `none`, `document_is_scheme`, `business_covered_by_scheme`, `reference_or_exclusion_only`, or `unclear`;
  - `kind`: `private_pool`, `statutory_pool`, `government_reinsurance_scheme`, `other`, or `unclear`;
  - `exact_name`: the exact pool or scheme name stated in the document, or null when it is not stated;
  - `jurisdiction_or_authority`: the named state, country, statute, fund, authority, or administrator when stated;
  - short line-referenced evidence.
- A short evidence item for each criterion, tied to numbered input lines, plus `certainty`: high/medium/low.
- A primary rejection reason chosen from a stable enum.

The local aggregator will set `qualifies=true` only for a complete/nearly-complete reinsurance contract with sufficient main terms, non-life business, treaty placement, and a non-government basis. Any `unclear` result will be non-qualifying for the automatic output and eligible for review or escalation.

Pool and scheme names must be extracted rather than normalized to a guessed name. A document that merely excludes or references a pool must not be classified as the pool itself. A missing/redacted counterparty is never, by itself, a reason for rejection or escalation.

## Document preparation

1. Join each yearly metadata CSV to its downloaded file by `downloadFilename`; record missing files rather than failing the run.
2. Decode text with replacement for damaged characters. For HTML, remove scripts, styles, inline XBRL noise, and hidden content while preserving headings, paragraphs, lists, and table cell boundaries.
3. Normalize whitespace, retain line breaks, and assign stable line numbers so model evidence can be checked against the submitted text.
4. Build an evidence pack instead of naively truncating the beginning or asking another LLM to summarize the file:
   - title, preamble, and table of contents;
   - the ending/signature and attachment area;
   - detected sections and deduplicated windows around headings and phrases for the reinsurance relationship, business covered, term, premium, limit/retention/attachment/share, exclusions, facultative/automatic/treaty status, amendment/endorsement language, pools, and statutory/government schemes;
   - document statistics and metadata.
5. Send full normalized text only when it is below the evidence budget. Otherwise allocate the budget by criterion so one long section cannot crowd out the other required terms. A sensible starting default is 24,000 characters, with an absolute ceiling of 32,000 characters including line labels and metadata.
6. Estimate tokens before every request with the best locally available tokenizer. When the exact tokenizer is unavailable, use a deliberately conservative character-to-token bound. Enforce both a configurable input-token ceiling and a separate output-token budget, leaving a context safety margin. Character caps remain in force even when a model advertises a much larger context window because cost, not only context, is a constraint.
7. If the first evidence pack leaves exactly one decisive criterion unresolved, a second call may use a small criterion-specific excerpt. Do not fall back to submitting the full document. Such calls count as escalations in the cost forecast.

The first implementation should avoid automatic lexical rejections except for unsupported/missing files. A high-recall local screen can be enabled later only if it has zero observed false negatives on the adjudicated set.

## Prompt design

The versioned prompt will:

- define all five criteria independently and emphasize that references to reinsurance are not enough;
- state that amendments, endorsements, extensions, commutations, services agreements, credit agreements, and corporate agreements are negatives even when their parties are insurers;
- distinguish treaty/automatic business from individual-risk facultative placements;
- explicitly exclude statutory funds and government-created schemes;
- distinguish non-life health/malpractice from life, annuity, pension, employee benefit, and health-plan business;
- allow confidential/redacted premium values while requiring evidence that a premium provision exists;
- explicitly state that missing or redacted counterparty names are acceptable and are not evidence of incompleteness;
- distinguish a contract that is itself a pool/government scheme from a contract that only covers, excludes, or references one, and return the exact stated scheme name and authority;
- instruct the model to use `unclear` rather than infer missing provisions;
- treat the document text as untrusted quoted material, not instructions;
- request only short, line-referenced evidence rather than free-form chain-of-thought.

Few-shot examples will be added only after the development set reveals repeatable errors. Examples used in the prompt will not appear in the holdout set.

## Evaluation and routing strategy

### Gold set

Create an adjudicated sample of about 160-200 documents, stratified across:

- clear complete treaties and retrocessions;
- near-complete contracts and contract forms;
- amendments, endorsements, extensions, and short letters;
- placement slips, cover notes, binders, and facultative/automatic edge cases;
- life, annuity, pension, health-plan, and ambiguous health documents;
- statutory government schemes;
- private pools, statutory pools, pool exclusions/references, and documents with redacted or missing counterparties;
- unrelated agreements that mention reinsurance;
- cases where the two historical classifiers disagree;
- a range of years, issuers, formats, and document sizes.

The labeling guide should explicitly settle whether placement slips/cover notes and unsigned contract forms count as “almost complete.” Until adjudicated, the conservative default is to separate them from automatic positives.

Split the set into a prompt-development portion and a blind holdout. Historical labels may select cases but may not determine the gold answer.

### Model benchmark

Run all three requested models on the same development evidence packs. Measure:

- positive-class precision, recall, F1, and confusion matrix;
- accuracy for each of the five criteria and each rejection category;
- abstention/unclear rate and structured-output retry/failure rate;
- latency, input/output tokens, and actual OpenRouter cost;
- exact-name accuracy for every relevant pool/government-scheme document, plus accuracy of `document_is_scheme` versus reference/exclusion-only involvement;
- results by document-length bin, including whether every decisive term survived evidence selection for the longest files.

Select the least expensive model that meets the quality threshold as the primary model. Route low-certainty results, rule/model contradictions, and borderline positives to the strongest second model. Use the third model only when the first two disagree on qualification or a decisive criterion. Consensus is taken per criterion; unresolved cases remain `manual_review`, not forced positives.

A useful initial acceptance target is at least 95% precision and 90% recall on the blind holdout, with results reported alongside confidence intervals and the review rate. The threshold can be adjusted if recall is more important than precision for the repository.

## Cost controls

- Normalize and build evidence packs for the candidate corpus locally before corpus-scale API work. Use their measured size distribution to forecast model input, rather than extrapolating from raw file sizes or a few hand-picked examples.
- Estimate request size before submission and reject any request that exceeds the evidence-pack or token cap.
- Record actual per-request provider, token counts, cost, latency, retry count, prompt version, and model ID.
- Add `--budget-usd`, with a conservative default below the available balance, and stop before the next request would exceed it. Budget accounting must include retries, targeted second calls, and model escalations.
- Make results append-only and resumable so interruptions never repeat completed calls.
- Keep concurrency small and configurable; retry only transient failures with bounded exponential backoff.
- Benchmark on the adjudicated sample before any corpus-scale run. Do not send all three models every document.
- After the pilot, forecast the remaining corpus cost from actual mean and high-percentile request cost, the observed escalation rate, retry rate, and number of unprocessed documents. Add at least a 20% contingency. Full classification is permitted only when this conservative forecast fits within the then-available OpenRouter credits and the CLI hard budget.
- If the forecast does not fit, stop after the benchmark/pilot and report the projected cost and the highest-value affordable subset. Do not silently reduce evidence quality or relax review/escalation rules to force the run under budget.
- Do not repeat smoke tests, catalog lookups, or model comparisons once they have answered their question. Prefer local parsing, mocked tests, and batched inspection before paid calls.

The earlier rough estimate suggested a primary pass might fit within $20, but it is not authorization for a full run. The pilot-derived conservative forecast and current remaining balance are the gating evidence.

## Intended deliverables

1. A small `src/` package for metadata loading, text extraction, evidence selection, model routing, aggregation, and resumable output.
2. Versioned prompt files and Pydantic schemas.
3. A CLI supporting year/file selection, `--limit`, dry run, model selection, concurrency, resume, `--max-input-chars`, `--max-input-tokens`, and a hard dollar budget ceiling.
4. CSV output for convenient filtering plus JSONL audit records containing criterion results and evidence.
5. Unit tests for parsing, evidence-window coverage/deduplication, classification invariants, budget enforcement, retry/resume behavior, and mocked provider routing.
6. A gold-label template, benchmark report, and a small real-document pilot report before a full run is authorized.

## Execution sequence

1. Finalize the labeling guide and adjudicate the edge-case sample.
2. Implement and test extraction/evidence packing locally with no API spend.
3. Implement the Pydantic AI classifier, prompt, provider lock, cost ledger, and resumable CLI.
4. Benchmark the three models on the development set and revise the prompt using error analysis.
5. Freeze the prompt and routing policy, then evaluate once on the holdout.
6. Run a 100-200 document pilot, inspect every predicted positive and a sample of negatives, and report cost/latency/error patterns.
7. Build the conservative remaining-corpus forecast, including the 20% contingency, and compare it with the current remaining credits and CLI budget.
8. Proceed to the full non-PDF corpus only if both quality and forecast gates pass. Otherwise stop with the pilot and recommendations.

## Handoff constraints and current state

This plan is intended for a fresh instance. It should continue from the established state rather than repeat discovery work.

- The repository inventory, representative document review, dependency installation, provider lookup, and first-party connectivity tests are already complete.
- `pyproject.toml` and `uv.lock` define the Python 3.13 environment. `scripts/diagnostics/check-openrouter.py` is a diagnostic utility, not a required preliminary step for future work.
- Preserve the pre-existing dirty worktree, particularly the renamed Gemini classification files. Do not clean up or rewrite unrelated changes.
- Do not use subagents unless a concrete, independent task will clearly reduce total usage. The default for this project is single-agent, local work.
- Minimize the instance's own tool and token usage: inspect targeted files, batch independent read-only checks, avoid reprinting large documents, and do not reload the full corpus into conversation context.
- Perform all extraction, evidence-pack construction, schema tests, and mocked routing tests locally before additional paid model calls.
- The next implementation task is the labeling guide plus local extractor/evidence-pack tests. It is not a full-corpus classification run.
- Counterparty disclosure is not a qualification condition. Pool/government-scheme identification and exact naming are required audit outputs.
- No full classification is authorized merely because code is ready; it remains conditional on the measured quality and conservative budget gates above.

## Implementation status (2026-08-30)

- The `src/reinsurance_classifier/` package implements metadata loading/deduplication, HTML/TXT normalization, stable line numbering, criterion-balanced evidence packs, strict Pydantic output, local qualification/consensus, first-party-only OpenRouter routes, bounded retry/escalation, hard budget reservations, append-only resume records, CSV output, the CLI, benchmark evaluation, candidate sampling, an independent-review/adjudication workflow, and the conservative full-run forecast gate.
- `docs/LABELING_GUIDE.md`, `prompts/classifier-v1.md`, `gold/gold-label-template.csv`, `gold/edge-case-seed.csv`, `gold/candidate-manifest.csv`, a leakage-free `gold/candidate-labeling.csv`, and the report templates are present.
- The full no-API preparation traversal completed for all 6,761 unique metadata filenames. It prepared all 6,678 supported non-PDF exhibits, recorded 64 missing files and 19 PDFs, and left zero errors. Measurements are in `reports/preparation-report.md`.
- The deterministic candidate manifest contains 180 documents split into 119 development and 61 holdout cases. It is a sampling manifest, not adjudicated gold.
- Independent human adjudication was unavailable, so the documented three-model/two-prompt silver protocol was used without treating its output as human gold. The original 180-document matrix resolved 155 cases; a fresh, untouched 60-document check resolved 49. The frozen production route matched all 49 resolved fresh cases (5 positives and 44 negatives), while 11 cases abstained.
- The conservative forecast gate permitted the run at a 20% contingency estimate of $17.1323 against the remaining authorization. The full append-only run then completed all 6,761 records: 651 qualified, 342 rejected, 5,685 manual review, 64 missing files, and 19 skipped PDFs, with zero remaining errors or budget-exhausted rows.
- A post-deployment 24-document, three-model/two-prompt accuracy check made 103 calls. Its strict quorum resolved 8 cases (6 positive, 2 negative); the production route matched all 8 and abstained from claiming truth for the other 16. This is supportive but small-sample autonomous-silver evidence.
- The provider-key meter reports $15.045276715 used under the $20 authorization, leaving $4.954723285. The local audit ledger is deliberately more conservative at $15.35171695 because failed/unknown-cost attempts are charged at their reserved amount.
- Final deliverables are `results/final/classification-latest.csv`, `results/final/classification-audit.jsonl`, and `reports/final-classification-report.md`. Automatic qualifications now also require at least one directly valid, internally consistent provider evidence trail; two legacy borderline positives were append-only corrected to manual review.
