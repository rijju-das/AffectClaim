"""Counterfactual scoring for person-specific affective claims."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from affectclaim.config import VerificationConfig
from affectclaim.domain.models import (
    CandidateClaim,
    Cue,
    CueFamily,
    EvidenceTestResult,
    ImageSample,
    InterventionKind,
    Person,
)
from affectclaim.interfaces.components import AffectEstimator, ClaimVerifier


class CounterfactualClaimVerifier(ClaimVerifier):
    def __init__(self, config: VerificationConfig) -> None:
        self._config = config

    def verify(
        self,
        sample: ImageSample,
        person: Person,
        cues: Sequence[Cue],
        claim: CandidateClaim,
        estimator: AffectEstimator,
    ) -> EvidenceTestResult:
        baseline = claim.affect_distribution.probabilities[claim.affect_label]
        removed = estimator.support_under_intervention(
            sample, person, cues, claim.affect_label, InterventionKind.REMOVE_EVIDENCE
        )
        isolated = estimator.support_under_intervention(
            sample, person, cues, claim.affect_label, InterventionKind.ISOLATE_EVIDENCE
        )
        nuisance = estimator.support_under_intervention(
            sample, person, cues, claim.affect_label, InterventionKind.NUISANCE
        )

        necessity = _clamp((baseline - removed.claim_support) / max(baseline, 1e-8))
        sufficiency = isolated.claim_support
        stability = _clamp(1.0 - abs(baseline - nuisance.claim_support))
        conflict = _cue_conflict(cues)
        score = (
            self._config.necessity_weight * necessity
            + self._config.sufficiency_weight * sufficiency
            + self._config.stability_weight * stability
            + self._config.conflict_weight * (1.0 - conflict)
        )
        return EvidenceTestResult(
            necessity=necessity,
            sufficiency=sufficiency,
            stability=stability,
            cue_conflict=conflict,
            evidence_score=_clamp(score),
            observations=(removed, isolated, nuisance),
        )


def _cue_conflict(cues: Sequence[Cue]) -> float:
    """Estimate cross-family disagreement from signed cue confidence.

    Adapters may describe suppressing evidence by prefixing a cue description with
    ``not:``. This transparent baseline convention will be replaced by explicit
    support/contradiction edges in a learned graph adapter.
    """

    family_scores: dict[CueFamily, list[float]] = defaultdict(list)
    for cue in cues:
        sign = -1.0 if cue.description.lower().startswith("not:") else 1.0
        family_scores[cue.family].append(sign * cue.confidence)
    means = [sum(values) / len(values) for values in family_scores.values()]
    if len(means) < 2:
        return 0.0
    positive = any(value > 0.1 for value in means)
    negative = any(value < -0.1 for value in means)
    if not (positive and negative):
        return 0.0
    return _clamp(max(means) - min(means))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
