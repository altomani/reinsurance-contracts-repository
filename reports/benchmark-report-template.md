# Classifier benchmark report

Prompt version: _TBD_  
Gold-set version: _TBD_  
Run ID and date: _TBD_

## Data and protocol

- Development documents: _TBD_
- Blind holdout documents: _TBD_
- Length/format coverage: _TBD_
- Models and first-party routes: _TBD_
- Evidence-pack limits: _TBD_

## Qualification performance

| Model/routing policy | Precision | Recall | F1 | 95% CI | Review rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| _TBD_ |  |  |  |  |  |

## Criterion and error analysis

Report accuracy for document/reinsurance, completeness, business basis, placement basis, and government basis. Include confusion matrices, rejection-reason accuracy, structured-output retry/failure rate, pool/scheme exact-name accuracy, and `document_is_scheme` versus reference/exclusion-only accuracy.

## Operational results

Report latency, input/output tokens, actual cost, retry rate, evidence coverage by normalized-length bin, and the observed escalation rate.

## Quality gate

- Holdout precision target (initial): 95%
- Holdout recall target (initial): 90%
- Result and confidence interval: _TBD_
- Prompt/routing policy frozen: _yes/no_

## Pilot and corpus forecast

After the 100–200 document pilot, report mean and high-percentile request cost, retries, escalations, unprocessed count, and a conservative remaining-corpus forecast with at least 20% contingency. Compare that forecast with both current credits and the CLI hard budget before authorizing a full run.
