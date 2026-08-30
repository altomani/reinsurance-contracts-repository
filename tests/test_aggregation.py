from __future__ import annotations

import pytest
from pydantic import ValidationError

from conftest import make_decision
from reinsurance_classifier.aggregation import (
    CriterionValue,
    OutcomeStatus,
    aggregate_decisions,
    criteria_disagree,
    needs_second_model,
)
from reinsurance_classifier.models import (
    BusinessBasis,
    Certainty,
    ClassificationDecision,
    DocumentKind,
    GovernmentBasis,
    RejectionReason,
    Completeness,
    MainTerms,
    TermAssessment,
    TermStatus,
    TriState,
    validate_evidence_lines,
    validate_evidence_quotes,
)


def test_all_five_gates_qualify_despite_missing_names_and_redacted_premium() -> None:
    result = aggregate_decisions([make_decision()])

    assert result.qualifies is True
    assert result.status == OutcomeStatus.QUALIFIED
    assert result.primary_rejection_reason == RejectionReason.NONE


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        (
            {
                "document_kind": DocumentKind.AMENDMENT_OR_ENDORSEMENT,
                "is_reinsurance_contract": TriState.YES,
            },
            RejectionReason.AMENDMENT_OR_ENDORSEMENT,
        ),
        (
            {"business_basis": BusinessBasis.LIFE_LIKE},
            RejectionReason.LIFE_LIKE_BUSINESS,
        ),
        (
            {"government_basis": GovernmentBasis.STATUTORY_GOVERNMENT_SCHEME},
            RejectionReason.STATUTORY_GOVERNMENT_SCHEME,
        ),
    ],
)
def test_any_failed_gate_rejects(
    changes: dict[str, object], reason: RejectionReason
) -> None:
    result = aggregate_decisions([make_decision(**changes)])

    assert result.qualifies is False
    assert result.status == OutcomeStatus.REJECTED
    assert result.primary_rejection_reason == reason


def test_unclear_gate_never_qualifies_and_requests_review() -> None:
    decision = make_decision(
        business_basis=BusinessBasis.UNCLEAR,
        certainty=Certainty.MEDIUM,
        primary_rejection_reason=RejectionReason.UNCLEAR_DECISIVE_CRITERION,
    )
    result = aggregate_decisions([decision])

    assert result.qualifies is False
    assert result.status == OutcomeStatus.MANUAL_REVIEW
    assert result.consensus.non_life_business == CriterionValue.UNCLEAR
    assert needs_second_model(decision) is True


def test_decisive_failure_precedes_an_unrelated_unclear_gate() -> None:
    decision = make_decision(
        document_kind=DocumentKind.AMENDMENT_OR_ENDORSEMENT,
        business_basis=BusinessBasis.UNCLEAR,
    )

    result = aggregate_decisions([decision])

    assert result.status == OutcomeStatus.MANUAL_REVIEW
    assert result.primary_rejection_reason == RejectionReason.AMENDMENT_OR_ENDORSEMENT


def test_explicit_failure_dominates_unclear_within_the_same_gate() -> None:
    document = make_decision(
        document_kind=DocumentKind.UNCLEAR,
        is_reinsurance_contract=TriState.NO,
    )
    base_terms = make_decision().main_terms
    incomplete = make_decision(
        main_terms=MainTerms(
            relationship_and_roles=TermAssessment(
                status=TermStatus.MISSING,
                evidence=base_terms.relationship_and_roles.evidence,
            ),
            business_covered=TermAssessment(
                status=TermStatus.UNCLEAR,
                evidence=base_terms.business_covered.evidence,
            ),
            term_or_period=base_terms.term_or_period,
            risk_transfer_economics=base_terms.risk_transfer_economics,
            premium_or_consideration=base_terms.premium_or_consideration,
            overall_completeness=Completeness.UNCLEAR,
        )
    )

    assert aggregate_decisions([document]).primary_rejection_reason == RejectionReason.NOT_REINSURANCE_CONTRACT
    assert aggregate_decisions([incomplete]).primary_rejection_reason == RejectionReason.INSUFFICIENT_MAIN_TERMS


def test_two_model_disagreement_is_unresolved_but_third_model_can_resolve() -> None:
    positive = make_decision()
    negative = make_decision(business_basis=BusinessBasis.LIFE_LIKE)

    assert criteria_disagree(positive, negative) is True
    assert aggregate_decisions([positive, negative]).status == OutcomeStatus.MANUAL_REVIEW
    resolved = aggregate_decisions([positive, negative, positive])
    assert resolved.qualifies is True


def test_provider_cannot_supply_qualifies_override() -> None:
    data = make_decision().model_dump(mode="json")
    data["qualifies"] = True

    with pytest.raises(ValidationError):
        ClassificationDecision.model_validate(data)


def test_evidence_must_reference_submitted_lines() -> None:
    decision = make_decision()
    validate_evidence_lines(decision, {1})

    with pytest.raises(ValueError, match="not submitted"):
        validate_evidence_lines(decision, {2, 3})


def test_evidence_quote_validation_accepts_grounded_normalized_text() -> None:
    validate_evidence_quotes(
        make_decision(), "[L000001] Operative   provision."
    )


def test_evidence_quote_validation_rejects_fabricated_text() -> None:
    with pytest.raises(ValueError, match="not grounded"):
        validate_evidence_quotes(
            make_decision(), "[L000001] Entirely different words"
        )
