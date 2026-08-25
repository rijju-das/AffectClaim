# Architecture

## Design principles

1. **Typed boundaries.** External models are translated into stable domain
   objects before their values enter the research pipeline.
2. **Dependency injection.** The pipeline receives component instances and does
   not construct models or use global state.
3. **Auditable outputs.** A `ClaimRecord` retains the person, cues, affect
   distribution, interventions, evidence score, decision, and final wording.
4. **Epistemic separation.** Recognition, verification, decision, and language
   realisation are different stages.
5. **Dataset neutrality.** No dataset-specific label set is embedded in the core.

## Core object relationships

```text
AffectClaimPipeline
  └── AffectClaimComponents
      ├── PersonGrounder
      ├── CueExtractor
      ├── AffectEstimator
      ├── ClaimGenerator
      ├── ClaimVerifier
      ├── DecisionPolicy
      └── LanguageRealiser
```

One class may implement multiple interfaces when this reflects the data source.
For example, `ManifestAdapter` reads all precomputed values from one immutable
manifest. A production composition should usually have separate adapters for
grounding, facial cues, body cues, context, and affect estimation.

## Baseline evidence score

The transparent baseline computes:

```text
E = w_n * necessity
  + w_s * sufficiency
  + w_t * stability
  + w_c * (1 - cue_conflict)
```

The weights are configuration rather than hidden constants. This baseline is not
a calibrated guarantee. A later conformal policy should be fitted on a held-out
calibration partition and evaluated using risk–coverage curves.
