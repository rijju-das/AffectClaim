"""Dependency-free JSON configuration loading."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VerificationConfig:
    necessity_weight: float = 0.35
    sufficiency_weight: float = 0.25
    stability_weight: float = 0.25
    conflict_weight: float = 0.15

    def __post_init__(self) -> None:
        total = self.necessity_weight + self.sufficiency_weight + self.stability_weight
        total += self.conflict_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Verification weights must sum to 1.0")


@dataclass(frozen=True, slots=True)
class DecisionConfig:
    direct_threshold: float = 0.72
    qualified_threshold: float = 0.48
    minimum_affect_probability: float = 0.40
    maximum_conflict_for_direct: float = 0.30

    def __post_init__(self) -> None:
        if self.qualified_threshold > self.direct_threshold:
            raise ValueError("Qualified threshold cannot exceed direct threshold")
        values = {
            "direct_threshold": self.direct_threshold,
            "qualified_threshold": self.qualified_threshold,
            "minimum_affect_probability": self.minimum_affect_probability,
            "maximum_conflict_for_direct": self.maximum_conflict_for_direct,
        }
        for key, value in values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class LanguageConfig:
    qualified_marker: str = "appears"


@dataclass(frozen=True, slots=True)
class AppConfig:
    verification: VerificationConfig = VerificationConfig()
    decision: DecisionConfig = DecisionConfig()
    language: LanguageConfig = LanguageConfig()

    @classmethod
    def from_json(cls, path: str | Path) -> AppConfig:
        with Path(path).open(encoding="utf-8") as handle:
            raw: Mapping[str, Any] = json.load(handle)
        return cls(
            verification=VerificationConfig(**raw.get("verification", {})),
            decision=DecisionConfig(**raw.get("decision", {})),
            language=LanguageConfig(**raw.get("language", {})),
        )
