"""Manifest-backed adapter for reproducible baselines and tests.

The adapter intentionally implements grounding, cue extraction, affect estimation,
and interventions as a single object because all values originate from one frozen
manifest. Production model integrations should normally implement these roles as
separate objects.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from affectclaim.domain.models import (
    AffectDistribution,
    BoundingBox,
    Cue,
    CueFamily,
    ImageSample,
    InterventionKind,
    InterventionObservation,
    Person,
)
from affectclaim.interfaces.components import AffectEstimator, CueExtractor, PersonGrounder


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    sample: ImageSample
    people: tuple[Person, ...]


class ManifestAdapter(PersonGrounder, CueExtractor, AffectEstimator):
    """Expose precomputed observations through the standard component APIs."""

    def __init__(self, records: Mapping[str, Mapping[str, Any]]) -> None:
        self._records = dict(records)

    @classmethod
    def from_json(cls, path: str | Path) -> ManifestAdapter:
        with Path(path).open(encoding="utf-8") as handle:
            content = json.load(handle)
        samples = content.get("samples")
        if not isinstance(samples, list):
            raise ValueError("Manifest must contain a 'samples' list")
        records = {str(item["sample_id"]): item for item in samples}
        if len(records) != len(samples):
            raise ValueError("Manifest sample IDs must be unique")
        return cls(records)

    def samples(self) -> list[ImageSample]:
        return [self._sample_from_record(record) for record in self._records.values()]

    def ground(self, sample: ImageSample) -> Sequence[Person]:
        record = self._record(sample)
        return tuple(self._person_from_dict(person) for person in record.get("people", []))

    def extract(self, sample: ImageSample, person: Person) -> Sequence[Cue]:
        raw_person = self._person_record(sample, person.person_id)
        return tuple(
            Cue(
                cue_id=str(cue["cue_id"]),
                person_id=person.person_id,
                family=CueFamily(str(cue["family"])),
                description=str(cue["description"]),
                confidence=float(cue["confidence"]),
                region=BoundingBox.from_sequence(cue["region"]) if cue.get("region") else None,
                source=str(cue.get("source", "manifest")),
            )
            for cue in raw_person.get("cues", [])
        )

    def estimate(
        self, sample: ImageSample, person: Person, cues: Sequence[Cue]
    ) -> AffectDistribution:
        del cues
        raw_person = self._person_record(sample, person.person_id)
        return AffectDistribution(
            {str(key): float(value) for key, value in raw_person["affect_distribution"].items()}
        )

    def support_under_intervention(
        self,
        sample: ImageSample,
        person: Person,
        cues: Sequence[Cue],
        affect_label: str,
        intervention: InterventionKind,
    ) -> InterventionObservation:
        del cues, affect_label
        raw_person = self._person_record(sample, person.person_id)
        interventions = raw_person.get("interventions", {})
        raw_value = interventions.get(intervention.value)
        if raw_value is None:
            raise ValueError(
                "Missing "
                f"{intervention.value!r} intervention for "
                f"{sample.sample_id}/{person.person_id}"
            )
        if isinstance(raw_value, list):
            support = sum(float(value) for value in raw_value) / len(raw_value)
            details: Mapping[str, Any] = {"replicates": raw_value}
        else:
            support = float(raw_value)
            details = {}
        return InterventionObservation(intervention, support, details)

    def _record(self, sample: ImageSample) -> Mapping[str, Any]:
        try:
            return self._records[sample.sample_id]
        except KeyError as error:
            raise KeyError(f"Unknown sample ID: {sample.sample_id}") from error

    def _person_record(self, sample: ImageSample, person_id: str) -> Mapping[str, Any]:
        for person in self._record(sample).get("people", []):
            if str(person["person_id"]) == person_id:
                return cast(Mapping[str, Any], person)
        raise KeyError(f"Unknown person ID {person_id!r} in sample {sample.sample_id!r}")

    @staticmethod
    def _sample_from_record(record: Mapping[str, Any]) -> ImageSample:
        raw_path = record.get("image_path")
        return ImageSample(
            sample_id=str(record["sample_id"]),
            image_path=Path(raw_path) if raw_path else None,
            metadata=record.get("metadata", {}),
        )

    @staticmethod
    def _person_from_dict(raw: Mapping[str, Any]) -> Person:
        return Person(
            person_id=str(raw["person_id"]),
            box=BoundingBox.from_sequence(raw["box"]),
            face_box=BoundingBox.from_sequence(raw["face_box"]) if raw.get("face_box") else None,
            confidence=float(raw.get("confidence", 1.0)),
        )
