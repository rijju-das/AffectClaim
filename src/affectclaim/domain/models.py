"""Domain objects for auditable affective claims.

The objects are model-agnostic by design. Adapters translate external model
outputs into these types, preventing framework code from depending on a specific
detector, emotion taxonomy, or language model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast


class CueFamily(str, Enum):
    FACE = "face"
    BODY = "body"
    CONTEXT = "context"
    OBJECT = "object"
    TEXT = "text"
    COLOUR = "colour"
    COMPOSITION = "composition"


class ClaimAction(str, Enum):
    DIRECT = "direct"
    QUALIFIED = "qualified"
    ABSTAIN = "abstain"


class InterventionKind(str, Enum):
    REMOVE_EVIDENCE = "remove_evidence"
    ISOLATE_EVIDENCE = "isolate_evidence"
    NUISANCE = "nuisance"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Normalised XYXY coordinates in the closed interval [0, 1]."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if any(value < 0.0 or value > 1.0 for value in values):
            raise ValueError("Bounding-box coordinates must be normalised to [0, 1]")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Bounding box must have positive width and height")

    @classmethod
    def from_sequence(cls, values: list[float] | tuple[float, ...]) -> BoundingBox:
        if len(values) != 4:
            raise ValueError("A bounding box requires four XYXY coordinates")
        return cls(*(float(value) for value in values))


@dataclass(frozen=True, slots=True)
class ImageSample:
    sample_id: str
    image_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Person:
    person_id: str
    box: BoundingBox
    face_box: BoundingBox | None = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        _validate_probability(self.confidence, "person confidence")


@dataclass(frozen=True, slots=True)
class Cue:
    cue_id: str
    person_id: str
    family: CueFamily
    description: str
    confidence: float
    region: BoundingBox | None = None
    source: str = "unknown"

    def __post_init__(self) -> None:
        _validate_probability(self.confidence, "cue confidence")


@dataclass(frozen=True, slots=True)
class AffectDistribution:
    probabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValueError("Affect distribution cannot be empty")
        if any(value < 0.0 for value in self.probabilities.values()):
            raise ValueError("Affect probabilities cannot be negative")
        total = sum(self.probabilities.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Affect probabilities must sum to 1.0, got {total:.6f}")

    @property
    def top_label(self) -> str:
        return max(self.probabilities, key=self.probabilities.__getitem__)

    @property
    def top_probability(self) -> float:
        return self.probabilities[self.top_label]


@dataclass(frozen=True, slots=True)
class CandidateClaim:
    claim_id: str
    person_id: str
    affect_label: str
    factual_description: str
    affect_distribution: AffectDistribution
    cited_cue_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InterventionObservation:
    kind: InterventionKind
    claim_support: float
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_probability(self.claim_support, "intervention claim support")


@dataclass(frozen=True, slots=True)
class EvidenceTestResult:
    necessity: float
    sufficiency: float
    stability: float
    cue_conflict: float
    evidence_score: float
    observations: tuple[InterventionObservation, ...] = ()

    def __post_init__(self) -> None:
        for name in ("necessity", "sufficiency", "stability", "cue_conflict", "evidence_score"):
            _validate_probability(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    sample_id: str
    person: Person
    cues: tuple[Cue, ...]
    candidate: CandidateClaim
    evidence: EvidenceTestResult
    action: ClaimAction
    realised_text: str

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _serialise(asdict(self)))


def _validate_probability(value: float, name: str) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be within [0, 1], got {value}")


def _serialise(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(item) for item in value]
    return value
