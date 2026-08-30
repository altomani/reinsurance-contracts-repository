# Autonomous validation protocol

This protocol is used when independent human labels are unavailable. Its output is explicitly called **silver**, not human gold.

1. Select the frozen 180-document stratified manifest (119 development, 61 holdout).
2. Run Qwen, DeepSeek, and GLM independently on every document with both the original criterion-first prompt and a separately worded fail-first prompt.
3. Validate every structured response locally. Schema and enum validation is mandatory. A citation is usable when either its full numbered span was submitted or its quoted words are grounded in the cited submitted lines; failure of both checks, or an internal rule contradiction, makes the vote hard-invalid. Line-span and quote-grounding rates are also reported separately.
4. Resolve each criterion only when at least four votes agree across all three models and both prompts. Derive qualification locally as the five-gate conjunction. A resolved failure rejects; unresolved documents remain `manual_review` and therefore never become automatic positives.
5. Estimate each model/prompt route against a model-excluded consensus: the comparison truth uses at least three agreeing votes from the other two providers across both prompts. Select a primary route on development data and report it once on the holdout.
6. Preserve every paid call, provider route, prompt version, token count, cost, evidence validation result, vote count, and unresolved case in append-only audit output.

For production routing, a semantic five-gate consensus is necessary but not sufficient for automatic qualification. At least one confirming provider call must also have a submitted line span or grounded quotation and no internal rule contradiction. Otherwise the record is conservatively downgraded to `manual_review`; the original call remains preserved in the append-only audit.

This design checks prompt sensitivity and provider agreement without pretending that correlated model judgments are independent human adjudication. Full-corpus use remains subject to the same 95% precision, 90% recall, cost-forecast, and hard-budget gates, with the report clearly identifying its autonomous-silver provenance.
