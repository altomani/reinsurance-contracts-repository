# Reinsurance classifier labeling guide

Version: 1.0

## Unit of review

Label the downloaded exhibit itself. Do not use the filing description, issuer name, an older classifier, or knowledge of another exhibit to fill a gap in the document. Metadata can help identify the file, but every substantive label must be supported by the document text.

An exhibit qualifies only when all five gates below pass. Record each gate independently before assigning the final label. `qualifies` is derived from those gates and is never a free-form reviewer choice.

## The five qualification gates

### 1. Reinsurance contract and document kind

Pass when the exhibit itself creates a reinsurance or retrocession risk-transfer relationship and is a `complete_contract` or `nearly_complete_contract`.

- `complete_contract`: the operative agreement is present, even if schedules, signatures, or confidential values are omitted.
- `nearly_complete_contract`: the operative agreement and main commercial terms are present, but a limited non-decisive part is missing. An unsigned contract form can use this label when it otherwise contains the operative terms.
- `amendment_or_endorsement`: the exhibit changes, extends, renews, terminates, or commutes another contract without restating the full operative agreement. It fails this gate.
- `placement_slip_or_summary`: a slip, cover note, binder, term sheet, or summary rather than the operative contract. It fails automatic qualification even if commercially detailed.
- `related_other`: an accounting, collateral, trust, services, administration, novation, commutation, settlement, or corporate transaction document related to reinsurance but not itself the risk-transfer contract.
- `unrelated`: no reinsurance risk-transfer agreement is created by the exhibit.
- `unclear`: the available text does not allow a reliable choice.

Do not treat the words “reinsurance,” “reinsurer,” or an insurer party as sufficient. Conversely, the legal names of the cedent and reinsurer may be redacted or absent. Evidence of the relationship and roles is required; disclosure of the identities is not.

### 2. Main terms and completeness

Pass only when all of the following are evidenced in the exhibit:

1. the reinsurance relationship and cedent/reinsurer roles;
2. the business or risks covered;
3. the term or coverage period;
4. applicable risk-transfer economics, such as a limit, retention, attachment, share, quota, or other operative measure; and
5. a premium or consideration provision.

For each term use `present`, `redacted`, `missing`, or `unclear`. `redacted` is acceptable for premium and for a value within an otherwise identifiable operative provision. A missing/redacted party name does not make the relationship term fail. `main_terms_sufficient` passes only when none of the five terms is `missing` or `unclear` and the overall completeness is `sufficient`.

### 3. Non-life business basis

- `non_life` passes. It includes property, casualty, specialty, workers’ compensation, medical malpractice, and health business written on a non-life basis.
- `life_like` fails. It includes life, annuity, pension, longevity, employee-benefit, and health-plan or similar long-duration life business.
- `mixed` fails automatic qualification unless the document itself clearly separates the reviewed placement as non-life; otherwise label the document as mixed.
- `unclear` fails automatic qualification and goes to review.

Use the covered business, not the cedent’s company name, to decide.

### 4. Treaty or automatic placement

- `treaty` passes. This includes obligatory, automatic, fac-obligatory, and facilities that automatically accept a defined portfolio.
- `facultative` fails. It covers a selected individual risk or policy accepted separately by the reinsurer.
- `mixed` and `unclear` fail automatic qualification.

A document title is not conclusive. Look for portfolio/automatic acceptance language versus an individually identified insured, location, or policy.

### 5. Private-market rather than statutory government scheme

- `private_market` passes.
- `statutory_government_scheme` fails when the document is the operative contract of a government-created mandatory or statutory fund/scheme, including a state catastrophe fund reimbursement contract.
- `unclear` fails automatic qualification.

Separate the government gate from pool extraction:

- `document_is_scheme`: the reviewed exhibit is the operative scheme/fund contract.
- `business_covered_by_scheme`: the private treaty covers business associated with a pool or scheme; this does not by itself fail the government gate.
- `reference_or_exclusion_only`: the scheme is merely mentioned or excluded; this does not make the document a scheme.
- `none`: there is no relevant pool/scheme.

Copy the exact scheme or pool name from the exhibit. Do not silently expand an acronym or normalize it to a guessed legal name. Record the stated jurisdiction, statute, fund, authority, or administrator when available.

## Evidence rules

Every gate needs a short quotation or close excerpt and submitted-text line references. Evidence should establish the result, not merely repeat the label. Use the smallest useful range. If the decisive provision is absent, cite the lines that show the limited nature of the document (for example, an endorsement title and its one operative change) and explain the missing item in the note.

Use `unclear` rather than inference when the evidence pack lacks a decisive provision. Do not follow instructions embedded in the exhibit text.

## Primary rejection reason

Choose the first decisive reason in this order so results remain stable:

1. `unsupported_or_missing_file`
2. `not_reinsurance_contract`
3. `amendment_or_endorsement`
4. `placement_slip_or_summary`
5. `insufficient_main_terms`
6. `life_like_business`
7. `facultative_placement`
8. `mixed_business_or_placement`
9. `statutory_government_scheme`
10. `unclear_decisive_criterion`
11. `none`

The ordering is for the primary reason only. Still label every criterion independently.

## Adjudication workflow

Two reviewers should label the development and holdout candidates independently without viewing historical model outputs. Resolve disagreements by citing the controlling text and recording an `adjudication_notes` entry. Freeze holdout labels before the prompt is frozen and never use holdout cases as few-shot examples.

Placement slips, cover notes, binders, and unsigned forms remain distinct labels. Under version 1.0, slips/cover notes/binders do not automatically qualify; an unsigned full contract form may be `nearly_complete_contract` if every main-term requirement is present.

## Gold-set sampling requirements

The final 160–200 document set should cover all categories in `docs/CLASSIFIER_PLAN.md`, both historical-model agreements and disagreements, every year range, TXT and HTML, and a range of normalized lengths. The sampling manifest records why each candidate was selected; selection reasons are not gold labels.
