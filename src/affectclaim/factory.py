"""Factories for standard pipeline compositions."""

from __future__ import annotations

from affectclaim.adapters.manifest import ManifestAdapter
from affectclaim.calibration.threshold import ThresholdDecisionPolicy
from affectclaim.config import AppConfig
from affectclaim.generation.template import TemplateClaimGenerator, TemplateLanguageRealiser
from affectclaim.pipeline import AffectClaimComponents, AffectClaimPipeline
from affectclaim.verification.counterfactual import CounterfactualClaimVerifier


def build_manifest_pipeline(adapter: ManifestAdapter, config: AppConfig) -> AffectClaimPipeline:
    return AffectClaimPipeline(
        AffectClaimComponents(
            grounder=adapter,
            cue_extractor=adapter,
            affect_estimator=adapter,
            claim_generator=TemplateClaimGenerator(),
            verifier=CounterfactualClaimVerifier(config.verification),
            decision_policy=ThresholdDecisionPolicy(config.decision),
            language_realiser=TemplateLanguageRealiser(config.language.qualified_marker),
        )
    )
