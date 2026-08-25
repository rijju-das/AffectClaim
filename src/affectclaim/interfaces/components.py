"""Abstract interfaces for all model-dependent pipeline stages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from affectclaim.domain.models import (
    AffectDistribution,
    CandidateClaim,
    ClaimAction,
    Cue,
    EvidenceTestResult,
    ImageSample,
    InterventionKind,
    InterventionObservation,
    Person,
)


class PersonGrounder(ABC):
    @abstractmethod
    def ground(self, sample: ImageSample) -> Sequence[Person]:
        """Locate people and associate their face and body regions."""


class CueExtractor(ABC):
    @abstractmethod
    def extract(self, sample: ImageSample, person: Person) -> Sequence[Cue]:
        """Return observable, person-linked cues rather than free-form reasoning."""


class AffectEstimator(ABC):
    @abstractmethod
    def estimate(
        self, sample: ImageSample, person: Person, cues: Sequence[Cue]
    ) -> AffectDistribution:
        """Estimate a distribution over apparent affect labels."""

    @abstractmethod
    def support_under_intervention(
        self,
        sample: ImageSample,
        person: Person,
        cues: Sequence[Cue],
        affect_label: str,
        intervention: InterventionKind,
    ) -> InterventionObservation:
        """Measure label support after a controlled intervention."""


class ClaimGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        sample: ImageSample,
        person: Person,
        cues: Sequence[Cue],
        distribution: AffectDistribution,
    ) -> CandidateClaim:
        """Construct a structured candidate; do not decide whether to report it."""


class ClaimVerifier(ABC):
    @abstractmethod
    def verify(
        self,
        sample: ImageSample,
        person: Person,
        cues: Sequence[Cue],
        claim: CandidateClaim,
        estimator: AffectEstimator,
    ) -> EvidenceTestResult:
        """Test whether the candidate depends on its cited evidence."""


class DecisionPolicy(ABC):
    @abstractmethod
    def decide(self, claim: CandidateClaim, evidence: EvidenceTestResult) -> ClaimAction:
        """Select direct, qualified, or abstained reporting."""


class LanguageRealiser(ABC):
    @abstractmethod
    def realise(self, claim: CandidateClaim, action: ClaimAction) -> str:
        """Render a controlled statement from an already-decided claim record."""
