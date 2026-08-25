"""Transparent baseline generation components."""

from __future__ import annotations

from collections.abc import Sequence

from affectclaim.domain.models import (
    AffectDistribution,
    CandidateClaim,
    ClaimAction,
    Cue,
    ImageSample,
    Person,
)
from affectclaim.interfaces.components import ClaimGenerator, LanguageRealiser


class TemplateClaimGenerator(ClaimGenerator):
    """Build candidates from structured values without hidden reasoning."""

    def generate(
        self,
        sample: ImageSample,
        person: Person,
        cues: Sequence[Cue],
        distribution: AffectDistribution,
    ) -> CandidateClaim:
        description = str(
            sample.metadata.get(
                f"description:{person.person_id}",
                sample.metadata.get(
                    "factual_description", f"A visible person ({person.person_id})"
                ),
            )
        ).rstrip(". ")
        cited = tuple(cue.cue_id for cue in cues if cue.confidence >= 0.5)
        return CandidateClaim(
            claim_id=f"{sample.sample_id}:{person.person_id}:{distribution.top_label}",
            person_id=person.person_id,
            affect_label=distribution.top_label,
            factual_description=description,
            affect_distribution=distribution,
            cited_cue_ids=cited,
        )


class TemplateLanguageRealiser(LanguageRealiser):
    """Realise epistemically controlled language for the baseline."""

    def __init__(self, qualified_marker: str = "appears") -> None:
        self._qualified_marker = qualified_marker.strip()

    def realise(self, claim: CandidateClaim, action: ClaimAction) -> str:
        factual = claim.factual_description.rstrip(". ")
        if action is ClaimAction.ABSTAIN:
            return f"{factual}."
        if action is ClaimAction.QUALIFIED:
            return f"{factual} and {self._qualified_marker} {claim.affect_label}."
        return f"{factual} and looks {claim.affect_label}."
