"""Qualification invariants and per-criterion model consensus."""

from __future__ import annotations

from collections import Counter
from enum import StrEnum

from pydantic import Field, model_validator

from .models import (
    BusinessBasis,
    Certainty,
    ClassificationDecision,
    Completeness,
    DocumentKind,
    GovernmentBasis,
    PASSING_DOCUMENT_KINDS,
    PlacementBasis,
    PoolInvolvement,
    RejectionReason,
    StrictModel,
    TermStatus,
    TriState,
)


class CriterionValue(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    UNCLEAR = "unclear"


class OutcomeStatus(StrEnum):
    QUALIFIED = "qualified"
    REJECTED = "rejected"
    MANUAL_REVIEW = "manual_review"


class CriterionConsensus(StrictModel):
    document_contract: CriterionValue
    main_terms: CriterionValue
    non_life_business: CriterionValue
    treaty_placement: CriterionValue
    private_market: CriterionValue

    def values(self) -> tuple[CriterionValue, ...]:
        return (
            self.document_contract,
            self.main_terms,
            self.non_life_business,
            self.treaty_placement,
            self.private_market,
        )


class AggregatedClassification(StrictModel):
    qualifies: bool
    status: OutcomeStatus
    consensus: CriterionConsensus
    primary_rejection_reason: RejectionReason
    model_count: int = Field(ge=1)
    direct_evidence_valid: bool = True

    @model_validator(mode="after")
    def validate_direct_evidence_gate(self) -> "AggregatedClassification":
        if self.qualifies and not self.direct_evidence_valid:
            raise ValueError("a qualified result requires directly valid evidence")
        return self


def decision_criteria(decision: ClassificationDecision) -> CriterionConsensus:
    if (
        decision.document_kind in PASSING_DOCUMENT_KINDS
        and decision.is_reinsurance_contract == TriState.YES
    ):
        document = CriterionValue.PASS
    elif (
        decision.document_kind not in PASSING_DOCUMENT_KINDS
        and decision.document_kind != DocumentKind.UNCLEAR
    ) or decision.is_reinsurance_contract == TriState.NO:
        document = CriterionValue.FAIL
    else:
        document = CriterionValue.UNCLEAR

    term_statuses = (
        decision.main_terms.relationship_and_roles.status,
        decision.main_terms.business_covered.status,
        decision.main_terms.term_or_period.status,
        decision.main_terms.risk_transfer_economics.status,
        decision.main_terms.premium_or_consideration.status,
    )
    if decision.main_terms.is_sufficient():
        main_terms = CriterionValue.PASS
    elif (
        decision.main_terms.overall_completeness == Completeness.INSUFFICIENT
        or TermStatus.MISSING in term_statuses
    ):
        main_terms = CriterionValue.FAIL
    else:
        main_terms = CriterionValue.UNCLEAR

    business = {
        BusinessBasis.NON_LIFE: CriterionValue.PASS,
        BusinessBasis.LIFE_LIKE: CriterionValue.FAIL,
        BusinessBasis.MIXED: CriterionValue.FAIL,
        BusinessBasis.UNCLEAR: CriterionValue.UNCLEAR,
    }[decision.business_basis]
    placement = {
        PlacementBasis.TREATY: CriterionValue.PASS,
        PlacementBasis.FACULTATIVE: CriterionValue.FAIL,
        PlacementBasis.MIXED: CriterionValue.FAIL,
        PlacementBasis.UNCLEAR: CriterionValue.UNCLEAR,
    }[decision.placement_basis]
    government = {
        GovernmentBasis.PRIVATE_MARKET: CriterionValue.PASS,
        GovernmentBasis.STATUTORY_GOVERNMENT_SCHEME: CriterionValue.FAIL,
        GovernmentBasis.UNCLEAR: CriterionValue.UNCLEAR,
    }[decision.government_basis]
    return CriterionConsensus(
        document_contract=document,
        main_terms=main_terms,
        non_life_business=business,
        treaty_placement=placement,
        private_market=government,
    )


def _consensus(votes: list[CriterionValue]) -> CriterionValue:
    if len(votes) == 1:
        return votes[0]
    counts = Counter(votes)
    value, count = counts.most_common(1)[0]
    if count >= len(votes) // 2 + 1:
        return value
    return CriterionValue.UNCLEAR


def aggregate_decisions(
    decisions: list[ClassificationDecision],
) -> AggregatedClassification:
    if not decisions:
        raise ValueError("at least one decision is required")
    per_model = [decision_criteria(decision) for decision in decisions]
    consensus = CriterionConsensus(
        document_contract=_consensus([item.document_contract for item in per_model]),
        main_terms=_consensus([item.main_terms for item in per_model]),
        non_life_business=_consensus([item.non_life_business for item in per_model]),
        treaty_placement=_consensus([item.treaty_placement for item in per_model]),
        private_market=_consensus([item.private_market for item in per_model]),
    )
    values = consensus.values()
    qualifies = all(value == CriterionValue.PASS for value in values)
    if CriterionValue.UNCLEAR in values:
        status = OutcomeStatus.MANUAL_REVIEW
    elif qualifies:
        status = OutcomeStatus.QUALIFIED
    else:
        status = OutcomeStatus.REJECTED
    return AggregatedClassification(
        qualifies=qualifies,
        status=status,
        consensus=consensus,
        primary_rejection_reason=_derive_rejection_reason(consensus, decisions),
        model_count=len(decisions),
    )


def _majority_enum(values: list[StrEnum]) -> StrEnum | None:
    counts = Counter(values)
    value, count = counts.most_common(1)[0]
    return value if count >= len(values) // 2 + 1 or len(values) == 1 else None


def _derive_rejection_reason(
    consensus: CriterionConsensus, decisions: list[ClassificationDecision]
) -> RejectionReason:
    if consensus.document_contract == CriterionValue.FAIL:
        kind = _majority_enum([decision.document_kind for decision in decisions])
        if kind == DocumentKind.AMENDMENT_OR_ENDORSEMENT:
            return RejectionReason.AMENDMENT_OR_ENDORSEMENT
        if kind == DocumentKind.PLACEMENT_SLIP_OR_SUMMARY:
            return RejectionReason.PLACEMENT_SLIP_OR_SUMMARY
        return RejectionReason.NOT_REINSURANCE_CONTRACT
    if consensus.main_terms == CriterionValue.FAIL:
        return RejectionReason.INSUFFICIENT_MAIN_TERMS
    if consensus.non_life_business == CriterionValue.FAIL:
        business = _majority_enum([decision.business_basis for decision in decisions])
        if business == BusinessBasis.MIXED:
            return RejectionReason.MIXED_BUSINESS_OR_PLACEMENT
        return RejectionReason.LIFE_LIKE_BUSINESS
    if consensus.treaty_placement == CriterionValue.FAIL:
        placement = _majority_enum([decision.placement_basis for decision in decisions])
        if placement == PlacementBasis.MIXED:
            return RejectionReason.MIXED_BUSINESS_OR_PLACEMENT
        return RejectionReason.FACULTATIVE_PLACEMENT
    if consensus.private_market == CriterionValue.FAIL:
        return RejectionReason.STATUTORY_GOVERNMENT_SCHEME
    if CriterionValue.UNCLEAR in consensus.values():
        return RejectionReason.UNCLEAR_DECISIVE_CRITERION
    return RejectionReason.NONE


def has_rule_contradiction(decision: ClassificationDecision) -> bool:
    """Find internal combinations that deserve a second-model check."""

    if (
        decision.document_kind in PASSING_DOCUMENT_KINDS
        and decision.is_reinsurance_contract != TriState.YES
    ):
        return True
    if (
        decision.main_terms.overall_completeness == Completeness.SUFFICIENT
        and not decision.main_terms.is_sufficient()
    ):
        return True
    if (
        decision.government_basis == GovernmentBasis.STATUTORY_GOVERNMENT_SCHEME
        and decision.pool_or_scheme.involvement == PoolInvolvement.NONE
    ):
        return True
    return False


def needs_second_model(decision: ClassificationDecision) -> bool:
    criteria = decision_criteria(decision)
    return (
        decision.certainty == Certainty.LOW
        or CriterionValue.UNCLEAR in criteria.values()
        or has_rule_contradiction(decision)
    )


def criteria_disagree(
    first: ClassificationDecision, second: ClassificationDecision
) -> bool:
    return decision_criteria(first).values() != decision_criteria(second).values()
