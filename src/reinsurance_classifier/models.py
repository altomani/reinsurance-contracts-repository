"""Validated model output and local audit schemas."""

from __future__ import annotations

from enum import StrEnum
import re
import unicodedata
from typing import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DocumentKind(StrEnum):
    COMPLETE_CONTRACT = "complete_contract"
    NEARLY_COMPLETE_CONTRACT = "nearly_complete_contract"
    AMENDMENT_OR_ENDORSEMENT = "amendment_or_endorsement"
    PLACEMENT_SLIP_OR_SUMMARY = "placement_slip_or_summary"
    RELATED_OTHER = "related_other"
    UNRELATED = "unrelated"
    UNCLEAR = "unclear"


class TriState(StrEnum):
    YES = "yes"
    NO = "no"
    UNCLEAR = "unclear"


class TermStatus(StrEnum):
    PRESENT = "present"
    REDACTED = "redacted"
    MISSING = "missing"
    UNCLEAR = "unclear"


class Completeness(StrEnum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    UNCLEAR = "unclear"


class BusinessBasis(StrEnum):
    NON_LIFE = "non_life"
    LIFE_LIKE = "life_like"
    MIXED = "mixed"
    UNCLEAR = "unclear"


class PlacementBasis(StrEnum):
    TREATY = "treaty"
    FACULTATIVE = "facultative"
    MIXED = "mixed"
    UNCLEAR = "unclear"


class CounterpartyDisclosure(StrEnum):
    NAMED = "named"
    PARTLY_REDACTED = "partly_redacted"
    FULLY_REDACTED = "fully_redacted"
    MISSING = "missing"
    UNCLEAR = "unclear"


class GovernmentBasis(StrEnum):
    PRIVATE_MARKET = "private_market"
    STATUTORY_GOVERNMENT_SCHEME = "statutory_government_scheme"
    UNCLEAR = "unclear"


class PoolInvolvement(StrEnum):
    NONE = "none"
    DOCUMENT_IS_SCHEME = "document_is_scheme"
    BUSINESS_COVERED_BY_SCHEME = "business_covered_by_scheme"
    REFERENCE_OR_EXCLUSION_ONLY = "reference_or_exclusion_only"
    UNCLEAR = "unclear"


class PoolKind(StrEnum):
    PRIVATE_POOL = "private_pool"
    STATUTORY_POOL = "statutory_pool"
    GOVERNMENT_REINSURANCE_SCHEME = "government_reinsurance_scheme"
    OTHER = "other"
    UNCLEAR = "unclear"


class Certainty(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RejectionReason(StrEnum):
    NONE = "none"
    UNSUPPORTED_OR_MISSING_FILE = "unsupported_or_missing_file"
    NOT_REINSURANCE_CONTRACT = "not_reinsurance_contract"
    AMENDMENT_OR_ENDORSEMENT = "amendment_or_endorsement"
    PLACEMENT_SLIP_OR_SUMMARY = "placement_slip_or_summary"
    INSUFFICIENT_MAIN_TERMS = "insufficient_main_terms"
    LIFE_LIKE_BUSINESS = "life_like_business"
    FACULTATIVE_PLACEMENT = "facultative_placement"
    MIXED_BUSINESS_OR_PLACEMENT = "mixed_business_or_placement"
    STATUTORY_GOVERNMENT_SCHEME = "statutory_government_scheme"
    UNCLEAR_DECISIVE_CRITERION = "unclear_decisive_criterion"


class EvidenceItem(StrictModel):
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=500)
    note: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_range(self) -> "EvidenceItem":
        if self.line_end < self.line_start:
            raise ValueError("line_end must be greater than or equal to line_start")
        return self


class TermAssessment(StrictModel):
    status: TermStatus
    evidence: EvidenceItem


class MainTerms(StrictModel):
    relationship_and_roles: TermAssessment
    business_covered: TermAssessment
    term_or_period: TermAssessment
    risk_transfer_economics: TermAssessment
    premium_or_consideration: TermAssessment
    overall_completeness: Completeness

    def is_sufficient(self) -> bool:
        terms = (
            self.relationship_and_roles,
            self.business_covered,
            self.term_or_period,
            self.risk_transfer_economics,
            self.premium_or_consideration,
        )
        acceptable = {TermStatus.PRESENT, TermStatus.REDACTED}
        return self.overall_completeness == Completeness.SUFFICIENT and all(
            term.status in acceptable for term in terms
        )

    def evidence_items(self) -> Iterable[EvidenceItem]:
        yield self.relationship_and_roles.evidence
        yield self.business_covered.evidence
        yield self.term_or_period.evidence
        yield self.risk_transfer_economics.evidence
        yield self.premium_or_consideration.evidence


class PoolOrScheme(StrictModel):
    involvement: PoolInvolvement
    kind: PoolKind
    exact_name: str | None = Field(default=None, max_length=300)
    jurisdiction_or_authority: str | None = Field(default=None, max_length=300)
    evidence: EvidenceItem

    @model_validator(mode="after")
    def validate_none_involvement(self) -> "PoolOrScheme":
        if self.involvement == PoolInvolvement.NONE and self.exact_name is not None:
            raise ValueError("exact_name must be null when pool involvement is none")
        return self


class CriterionEvidence(StrictModel):
    reinsurance_and_document_kind: EvidenceItem
    completeness: EvidenceItem
    business_basis: EvidenceItem
    placement_basis: EvidenceItem
    government_basis: EvidenceItem

    def items(self) -> Iterable[EvidenceItem]:
        yield self.reinsurance_and_document_kind
        yield self.completeness
        yield self.business_basis
        yield self.placement_basis
        yield self.government_basis


class ClassificationDecision(StrictModel):
    """Provider output. Deliberately excludes a model-controlled qualifies flag."""

    document_kind: DocumentKind
    is_reinsurance_contract: TriState
    main_terms: MainTerms
    business_basis: BusinessBasis
    placement_basis: PlacementBasis
    counterparty_disclosure: CounterpartyDisclosure
    government_basis: GovernmentBasis
    pool_or_scheme: PoolOrScheme
    criterion_evidence: CriterionEvidence
    certainty: Certainty
    primary_rejection_reason: RejectionReason

    def all_evidence_items(self) -> Iterable[EvidenceItem]:
        yield from self.main_terms.evidence_items()
        yield self.pool_or_scheme.evidence
        yield from self.criterion_evidence.items()


PASSING_DOCUMENT_KINDS = {
    DocumentKind.COMPLETE_CONTRACT,
    DocumentKind.NEARLY_COMPLETE_CONTRACT,
}


def validate_evidence_lines(
    decision: ClassificationDecision, selected_line_numbers: set[int]
) -> None:
    """Reject evidence that points outside the exact submitted exhibit lines."""

    for evidence in decision.all_evidence_items():
        referenced = set(range(evidence.line_start, evidence.line_end + 1))
        if not referenced.issubset(selected_line_numbers):
            missing = sorted(referenced - selected_line_numbers)
            raise ValueError(f"evidence references lines not submitted: {missing[:5]}")


_PACK_LINE = re.compile(r"^\[L(\d{6})\]\s?(.*)$")
_GROUNDING_TOKEN = re.compile(r"[a-z0-9]+")


def validate_evidence_quotes(decision: ClassificationDecision, pack_text: str) -> None:
    """Reject citations whose quoted words are not grounded in their line span."""

    lines: dict[int, str] = {}
    for raw_line in pack_text.splitlines():
        match = _PACK_LINE.match(raw_line)
        if match:
            lines[int(match.group(1))] = match.group(2)
    for evidence in decision.all_evidence_items():
        span = " ".join(
            lines[number]
            for number in range(evidence.line_start, evidence.line_end + 1)
            if number in lines
        )
        quote = _normalize_grounding_text(evidence.quote)
        source = _normalize_grounding_text(span)
        if quote and quote in source:
            continue
        quote_tokens = _GROUNDING_TOKEN.findall(quote)
        source_tokens = set(_GROUNDING_TOKEN.findall(source))
        substantive = [token for token in quote_tokens if len(token) >= 3]
        coverage = (
            sum(token in source_tokens for token in substantive) / len(substantive)
            if substantive
            else 0.0
        )
        if coverage < 0.9:
            raise ValueError(
                "evidence quote is not grounded in referenced lines "
                f"L{evidence.line_start:06d}-L{evidence.line_end:06d}"
            )


def _normalize_grounding_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("…", " ")
    value = re.sub(r"\.{3,}", " ", value)
    return " ".join(value.split()).strip(" \"'`.,;:!?()[]{}")
