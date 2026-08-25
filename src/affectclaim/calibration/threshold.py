"""Threshold baseline for direct, qualified, and abstained reporting."""

from __future__ import annotations

from affectclaim.config import DecisionConfig
from affectclaim.domain.models import CandidateClaim, ClaimAction, EvidenceTestResult
from affectclaim.interfaces.components import DecisionPolicy


class ThresholdDecisionPolicy(DecisionPolicy):
    """Deterministic baseline to be superseded by fitted conformal policies."""

    def __init__(self, config: DecisionConfig) -> None:
        self._config = config

    def decide(self, claim: CandidateClaim, evidence: EvidenceTestResult) -> ClaimAction:
        if claim.affect_distribution.top_probability < self._config.minimum_affect_probability:
            return ClaimAction.ABSTAIN
        if (
            evidence.evidence_score >= self._config.direct_threshold
            and evidence.cue_conflict <= self._config.maximum_conflict_for_direct
        ):
            return ClaimAction.DIRECT
        if evidence.evidence_score >= self._config.qualified_threshold:
            return ClaimAction.QUALIFIED
        return ClaimAction.ABSTAIN
