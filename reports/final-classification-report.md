# Final autonomous classification report

Date: 2026-08-30  
Prompt: `classifier-v1`  
Label provenance: autonomous silver, not independent human gold

## Final corpus outcome

The resumable run reached a terminal latest state for all 6,761 unique metadata filenames. No record remains in `error` or `budget_exhausted`.

| Latest status | Count |
| --- | ---: |
| Qualified | 651 |
| Rejected | 342 |
| Manual review | 5,685 |
| Missing file | 64 |
| Skipped PDF | 19 |
| **Total** | **6,761** |

The latest CSV has exactly 6,761 data rows. The 53 MB append-only JSONL contains 6,838 audit records, including recovery retries and two final conservative corrections. It preserves all earlier decisions rather than rewriting history.

## Routing and evidence integrity

The final latest state contains 13,014 paid provider calls:

| Model / required first-party provider | Calls |
| --- | ---: |
| GLM 5.3 Flash / Z.AI | 6,675 |
| Qwen 3.8 Flash / Alibaba | 6,336 |
| DeepSeek V4 Flash Vision / DeepSeek | 3 |

Provider fallbacks were disabled. Of the latest calls, 93.57% had either a valid submitted line span or a grounded quotation; 91.49% also had no internal rule contradiction. Quote grounding alone was 76.19%.

Qualification is deliberately stricter than a model's own label. Python computes the five semantic gates, the route confirms positives with a second provider, and at least one provider must supply directly valid evidence without an internal rule contradiction. The final audit found no remaining qualified record without such a provider. Two prior borderline qualifications failed this last condition and were append-only downgraded to `manual_review`.

## Accuracy checks

The primary fresh check used 60 documents excluded from the original 180-document sample, three models, and two independently worded prompts. Strict supermajority concordance resolved 49 cases: 5 qualified and 44 rejected; 11 abstained. The frozen production route matched all 49 resolved silver labels (TP=5, TN=44, FP=0, FN=0). Point precision and recall were both 100%; the two-sided Wilson lower bound was 56.55% because there were only five resolved positives.

After the corpus run, a second small check sampled 24 previously unseen completed documents. It made 103 model/prompt calls before its small audit budget was reached. Strict concordance resolved 8 cases (6 positive, 2 negative) and abstained on 16. The production route matched all 8 resolved labels (TP=6, TN=2, FP=0, FN=0); point precision and recall were both 100%, with a 60.97% Wilson lower bound.

These checks test provider concordance, prompt sensitivity, structured-output validity, and route behavior. They do not replace independent legal or human adjudication. Their conservative response to insufficient agreement is `manual_review`, which explains the high 84.09% review rate.

Supporting artifacts:

- `reports/autonomous-validation.json`
- `reports/fresh-autonomous-validation.json`
- `reports/fresh-holdout-evaluation.json`
- `reports/postdeploy-accuracy-audit.json`
- `reports/postdeploy-route-accuracy.json`
- `reports/final-integrity.json`
- `gold/autonomous-silver-labels.csv`
- `gold/fresh-autonomous-silver-labels.csv`
- `gold/postdeploy-autonomous-silver-labels.csv`

## Budget reconciliation

The authorized ceiling was $20. The OpenRouter provider-key meter reports **$15.045276715** used and **$4.954723285** remaining. The repository's conservative audit ledger totals **$15.35171695** because transient failures and calls without returned cost are charged at the full reservation. Both measures remain within authorization; the provider meter is the authoritative billed amount.

The pre-run gate had forecast $17.13225204 including 20% contingency, so actual metered usage finished $2.086975325 below that conservative forecast and $4.954723285 below the authorization.

## Deliverables and verification

- Latest classifications: `results/final/classification-latest.csv`
- Complete append-only audit: `results/final/classification-audit.jsonl`
- Frozen full-run gate: `reports/full-run-gate.json`
- Machine-readable final reconciliation: `reports/final-integrity.json`
- Autonomous validation protocol: `docs/AUTONOMOUS_VALIDATION.md`

Final local verification: 43 tests passed, the lock file is current, the CSV and JSONL reconcile to 6,761 latest records, no automatic positive lacks directly valid provider evidence, and no latest record is an error or budget exhaustion.
