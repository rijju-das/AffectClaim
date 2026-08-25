"""AffectClaim public package API."""

from affectclaim.domain.models import (
    AffectDistribution,
    BoundingBox,
    ClaimAction,
    ClaimRecord,
    Cue,
    CueFamily,
    ImageSample,
    Person,
)
from affectclaim.pipeline import AffectClaimComponents, AffectClaimPipeline

__all__ = [
    "AffectClaimComponents",
    "AffectClaimPipeline",
    "AffectDistribution",
    "BoundingBox",
    "ClaimAction",
    "ClaimRecord",
    "Cue",
    "CueFamily",
    "ImageSample",
    "Person",
]

__version__ = "0.1.0"
