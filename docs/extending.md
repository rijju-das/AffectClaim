# Extending AffectClaim

## Adding a vision model

Implement one or more interfaces from `affectclaim.interfaces.components`.
Adapters should:

- load expensive weights once in their constructor;
- accept device and checkpoint configuration explicitly;
- return normalised boxes and probabilities;
- preserve the external model name/version in each cue's `source` field;
- avoid unrestricted rationales when a typed observation is available;
- raise a clear exception when an intervention cannot be performed.

Suggested adapters are:

```text
GroundingAdapter      -> PersonGrounder
ActionUnitAdapter     -> CueExtractor
BodyPoseAdapter       -> CueExtractor
SceneContextAdapter   -> CueExtractor
GraphAffectEstimator  -> AffectEstimator
ConformalPolicy       -> DecisionPolicy
```

Multiple cue extractors can be combined later through a `CompositeCueExtractor`
without changing the pipeline contract.

## Adding an emotion taxonomy

The core stores labels as strings and therefore does not impose Ekman, EMOTIC, or
valence–arousal categories. A dataset adapter is responsible for mapping native
annotations into the declared experiment taxonomy. Store that mapping as
versioned configuration and never silently merge person emotion, scene sentiment,
and viewer-evoked emotion.

## Adding a fitted reporting policy

Implement `DecisionPolicy.decide`. A fitted policy should save only lightweight
parameters and calibration metadata in Git; sample-level calibration data and
binary estimators belong in ignored directories. Record:

- calibration split identifier;
- target risk or coverage;
- nonconformity score definition;
- fitted threshold;
- assumptions and known distribution shifts.

## Reproducibility contract

Every experiment should retain:

- source revision;
- configuration;
- dataset and split identifiers;
- model/checkpoint identifiers without copying weights;
- random seeds;
- structured claim records;
- aggregate metrics and environment information.
