# AffectClaim

Counterfactual evidence calibration for selective visual affect reporting.

AffectClaim asks a narrower question than conventional emotion-aware captioning:
**is an affective statement visually supported strongly enough to be made?** The
pipeline associates claims with a person, records observable evidence, tests the
claim under controlled interventions, and chooses whether to state, qualify, or
omit the affective phrase.

## Current scope

This repository contains the research framework and a runnable manifest-based
baseline. Heavy vision models are deliberately kept behind interfaces so that
grounders, Action Unit detectors, affect estimators, and language models can be
changed without rewriting the pipeline.

```text
image / manifest
  -> person grounding
  -> observable cue extraction
  -> affect estimation
  -> candidate claim generation
  -> counterfactual evidence verification
  -> direct / qualified / abstained reporting
```

The baseline consumes precomputed observations from JSON. It is useful for
testing the claim schema, calibration policy, annotation process, and evaluation
code before expensive model adapters are introduced.

## Quick start

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
affectclaim run --manifest examples/sample_manifest.json --output outputs/sample.json
pytest
```

The command prints the realised reports and writes complete auditable claim
records to `outputs/sample.json`.

## Repository layout

```text
src/affectclaim/
  domain/          typed research entities
  interfaces/      replaceable component contracts
  adapters/        data/model adapters
  verification/    counterfactual evidence tests
  calibration/     selective reporting policies
  generation/      controlled language realisation
  pipeline.py      orchestration through dependency injection
configs/           versioned lightweight configuration
examples/          tiny versioned examples only
docs/              research design and extension guidance
scripts/           repository/data safety utilities
tests/             unit and integration tests
data/              local datasets (ignored except documentation)
checkpoints/       local model weights (ignored)
outputs/           generated runs (ignored)
```

## Extending the system

Implement the relevant abstract interface in
`src/affectclaim/interfaces/components.py`, then supply the object through
`AffectClaimComponents`. No global model state is used. A model adapter can
therefore be tested independently and exchanged through configuration or a
factory in a future experiment layer.

The framework deliberately distinguishes:

- observable cues from affective interpretation;
- a person's apparent expression from their internal emotional state;
- person-specific affect from scene- or viewer-evoked affect;
- model confidence from counterfactual evidence support.

See [the full research concept](docs/research_concept.md) and
[extension guidance](docs/extending.md).

## Data and model policy

Datasets, checkpoints, generated outputs, experiment trackers, archives, and
common binary model formats are ignored. Do not force-add them to Git. Before a
commit, run:

```bash
python scripts/check_repository.py
```

Metadata, manifests without private paths, annotation schemas, and small test
fixtures should remain versioned for reproducibility.
