from __future__ import annotations

from reinsurance_classifier.models import (
    BusinessBasis,
    Certainty,
    ClassificationDecision,
    Completeness,
    CounterpartyDisclosure,
    CriterionEvidence,
    DocumentKind,
    EvidenceItem,
    GovernmentBasis,
    MainTerms,
    PlacementBasis,
    PoolInvolvement,
    PoolKind,
    PoolOrScheme,
    RejectionReason,
    TermAssessment,
    TermStatus,
    TriState,
)


def evidence(line: int = 1, quote: str = "operative provision") -> EvidenceItem:
    return EvidenceItem(
        line_start=line,
        line_end=line,
        quote=quote,
        note="supports the criterion",
    )


def make_decision(**changes: object) -> ClassificationDecision:
    item = evidence()
    term = TermAssessment(status=TermStatus.PRESENT, evidence=item)
    data: dict[str, object] = {
        "document_kind": DocumentKind.COMPLETE_CONTRACT,
        "is_reinsurance_contract": TriState.YES,
        "main_terms": MainTerms(
            relationship_and_roles=term,
            business_covered=term,
            term_or_period=term,
            risk_transfer_economics=term,
            premium_or_consideration=TermAssessment(
                status=TermStatus.REDACTED, evidence=item
            ),
            overall_completeness=Completeness.SUFFICIENT,
        ),
        "business_basis": BusinessBasis.NON_LIFE,
        "placement_basis": PlacementBasis.TREATY,
        "counterparty_disclosure": CounterpartyDisclosure.MISSING,
        "government_basis": GovernmentBasis.PRIVATE_MARKET,
        "pool_or_scheme": PoolOrScheme(
            involvement=PoolInvolvement.NONE,
            kind=PoolKind.OTHER,
            exact_name=None,
            jurisdiction_or_authority=None,
            evidence=item,
        ),
        "criterion_evidence": CriterionEvidence(
            reinsurance_and_document_kind=item,
            completeness=item,
            business_basis=item,
            placement_basis=item,
            government_basis=item,
        ),
        "certainty": Certainty.HIGH,
        "primary_rejection_reason": RejectionReason.NONE,
    }
    data.update(changes)
    return ClassificationDecision.model_validate(data)
