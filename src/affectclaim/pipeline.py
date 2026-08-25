"""Object-oriented AffectClaim pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from affectclaim.domain.models import ClaimRecord, ImageSample
from affectclaim.interfaces.components import (
    AffectEstimator,
    ClaimGenerator,
    ClaimVerifier,
    CueExtractor,
    DecisionPolicy,
    LanguageRealiser,
    PersonGrounder,
)


@dataclass(frozen=True, slots=True)
class AffectClaimComponents:
    grounder: PersonGrounder
    cue_extractor: CueExtractor
    affect_estimator: AffectEstimator
    claim_generator: ClaimGenerator
    verifier: ClaimVerifier
    decision_policy: DecisionPolicy
    language_realiser: LanguageRealiser


class AffectClaimPipeline:
    """Coordinates components while keeping their implementations replaceable."""

    def __init__(self, components: AffectClaimComponents) -> None:
        self._components = components

    def analyse(self, sample: ImageSample) -> list[ClaimRecord]:
        records: list[ClaimRecord] = []
        for person in self._components.grounder.ground(sample):
            cues = tuple(self._components.cue_extractor.extract(sample, person))
            distribution = self._components.affect_estimator.estimate(sample, person, cues)
            candidate = self._components.claim_generator.generate(
                sample, person, cues, distribution
            )
            evidence = self._components.verifier.verify(
                sample,
                person,
                cues,
                candidate,
                self._components.affect_estimator,
            )
            action = self._components.decision_policy.decide(candidate, evidence)
            text = self._components.language_realiser.realise(candidate, action)
            records.append(
                ClaimRecord(
                    sample_id=sample.sample_id,
                    person=person,
                    cues=cues,
                    candidate=candidate,
                    evidence=evidence,
                    action=action,
                    realised_text=text,
                )
            )
        return records
