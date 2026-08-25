# AffectClaim: Counterfactual Evidence Calibration for Selective Visual Affect Reporting

## Recommendation

The strongest new direction is not another affect-aware caption generator. It is a system and benchmark for deciding **whether an affective claim is visually supportable at all**.

This distinction follows directly from the thesis. The thesis repeatedly separates observable facial or scene evidence from an inference about affect, identifies ambiguity and annotation mismatch as persistent limitations, and argues that fluent language is not itself evidence. The earlier captioning study showed that affect can enrich a description, but it also exposed dominant-face errors, categorical overconfidence, limited affective vocabulary, and weak evaluation of emotional correctness. The later Action Unit, scene-intervention, SS-VLM, and retrieval-grounding studies provide the components needed to address those weaknesses without simply replacing VGG16/LSTM with a newer backbone.

The proposed paper is therefore:

> **AffectClaim: Counterfactual Evidence Calibration for Selective Visual Affect Reporting**

Alternative titles:

- **Should the Model Say It? Counterfactually Verified Affective Claims from Images**
- **From Plausible to Supported: Risk-Controlled Visual Affect Reporting**
- **Who Appears to Feel What, and Is the Evidence Sufficient?**

## Central research question

Can a vision--language system distinguish an observable cue from an affective interpretation, associate each interpretation with the correct person or scene region, and abstain when counterfactual tests show that the cited evidence is insufficient or unstable?

The intended progression is:

```text
image
  -> people and scene regions
  -> observable cue graph
  -> candidate affective claims
  -> counterfactual evidence tests
  -> calibrated claim selection or abstention
  -> factual description with qualified affective language
```

The primary output is not a free-form chain of thought. It is an auditable claim record:

```json
{
  "subject": "person_2",
  "subject_box": [0.42, 0.18, 0.71, 0.91],
  "observable_cues": [
    {"type": "face", "cue": "lip-corner raising", "region": "face_2"},
    {"type": "body", "cue": "raised arms", "region": "body_2"},
    {"type": "context", "cue": "crossing a finish line", "region": "scene_1"}
  ],
  "affect_distribution": {"positive": 0.62, "neutral_or_unclear": 0.30, "negative": 0.08},
  "claim": "Person 2 appears pleased after crossing the finish line.",
  "evidence_necessity": 0.71,
  "nuisance_stability": 0.83,
  "calibrated_action": "qualified_claim"
}
```

The language must preserve the epistemic boundary: the model reports what a person **appears** to express from visible evidence; it does not assert an internal emotional state.

## Why this is stronger than the supplied captioning proposal

The supplied vision remains well aligned with the thesis, particularly its person grounding, uncertainty, and counterfactual testing. Its main novelty claim is now exposed, however, by **SEA-Cap** (Cai et al., arXiv:2607.25789, 28 July 2026), which already uses local sentimental evidence mining, object-level grounding, caption generation, and hallucination verification. A paper centred on “evidence-grounded affective captioning” would therefore face a direct and very recent comparison.

AffectClaim moves the contribution from generation quality to **claim admissibility**:

- SEA-Cap asks how a requested sentiment can be expressed faithfully in a caption.
- AffectClaim asks whether an affective statement is supported strongly enough to be made.
- SEA-Cap verifies captions against mined evidence.
- AffectClaim tests the claimed evidence through removal, isolation, cue conflict, and nuisance-preserving interventions.
- AffectClaim explicitly separates person attribution, observer disagreement, evidence sufficiency, and selective abstention.

This is also different from generic counterfactual explanation. The target is not merely whether masking a region changes a class score. The target is a structured, person-specific affective claim whose evidence and linguistic certainty must agree.

## Proposed contributions

### 1. A person--cue--claim graph

Represent an image as a typed graph with:

- person nodes and associated face/body regions;
- observable cue nodes: Action Units, gaze, pose, gesture, interaction, objects, scene context, text, colour, and composition;
- candidate affect nodes represented as distributions rather than single labels;
- support, contradiction, and attribution edges linking evidence to a specific claim.

The graph provides an inspectable interface between specialist visual models and the language model. It also prevents evidence about one person from being silently used to describe another.

### 2. Counterfactual evidence calibration

For a candidate claim `c` and cited region set `R`, evaluate four properties:

1. **Necessity:** Does support for `c` decrease when `R` is removed or neutralised?
2. **Restricted sufficiency:** Does `R`, when shown with the minimum context required to interpret it, retain useful support for `c`?
3. **Cue conflict:** Do face, body, and context experts disagree in a way that should widen the affect distribution or trigger abstention?
4. **Nuisance stability:** Does the claim remain stable under perturbations that should not change affect, such as mild crop, compression, brightness variation, or identity-preserving background changes?

A possible evidence score is:

```text
E(c,R) = w_n * Necessity(c,R)
       + w_s * Sufficiency(c,R)
       + w_t * Stability(c)
       - w_x * CueConflict(c)
```

This score should not be presented as causal proof of human emotion. It measures dependence of a model claim on a controlled set of visible inputs.

### 3. Risk-controlled selective affect reporting

Calibrate the decision to state, qualify, or omit an affective claim on a held-out calibration set. Split conformal prediction or conformal risk control can turn the evidence score and affect distribution into a decision rule:

- **state:** evidence is strong and annotator agreement is high;
- **qualify:** evidence is useful but ambiguous, producing wording such as “appears pleased”;
- **abstain:** evidence is insufficient, conflicting, or unstable, so the output remains factual.

The formal guarantee must be stated narrowly. Under the required exchangeability assumptions, calibration can control a pre-defined claim-error or coverage criterion. It cannot guarantee that an emotion inference is psychologically true.

### 4. AffectClaim-Bench

Create a focused evaluation set rather than claiming a universal affect dataset. A practical first version could contain 1,000--1,500 images, sampled into five diagnostic strata:

- face-dominant evidence;
- body- or action-dominant evidence;
- context-dominant evidence;
- multiple people with different apparent affect;
- ambiguous, conflicting, occluded, or insufficient evidence.

Each image should have repeated human annotations for:

- target person or scene;
- applicable affect labels and valence--arousal ratings;
- visible cue families supporting the judgement;
- evidence region(s);
- ambiguity or “insufficient evidence”;
- acceptable language strength: direct, qualified, or omit;
- a factual caption and, where supported, an affect-aware caption.

EMOTIC is the most natural source for person-context images because it includes person boxes, apparent-emotion categories, and dimensional annotations. EmoSet, FindingEmo, or ArtEmis can supply separate scene/evoked-affect test strata, but their labels must not be collapsed into person emotion. Flickr30K or COCO can be used only for factual caption evaluation. Dataset-specific targets should remain separate tasks joined by a shared claim schema.

## Model design

### Stage A: spatial and person grounding

Use Florence-2, Grounding DINO plus SAM, or Qwen3-VL grounding to identify people, faces, bodies, objects, and relevant scene regions. Explicitly link each face to its body and person identifier. Images with uncertain face--body association should be marked rather than resolved by largest-face heuristics.

### Stage B: cue experts

Reuse thesis-aligned specialists where they add measurable information:

- SS-VLM/OpenFace-derived facial Action Units and expression evidence;
- valence--arousal estimates as graded facial evidence;
- body pose and interaction features;
- scene/object/text/colour experts developed through AdMoteNet, the palette work, or CAAR;
- a general open VLM such as Qwen3-VL-4B/8B for spatial and relational context.

These experts should output typed observations and calibrated scores, not unrestricted rationales.

### Stage C: structured fusion

A graph transformer or compact gated mixture-of-experts fuses person and cue nodes. The router is supervised by cue annotations and perturbation consistency. Its purpose is not simply to improve emotion classification; it predicts which evidence families support each person-specific claim and exposes disagreement between them.

### Stage D: claim generation and verification

Generate claims from the structured graph under a constrained schema. Re-score each claim after evidence deletion, evidence isolation, and nuisance interventions. Reject claims that cite the wrong subject, remain unchanged when their alleged evidence is removed, or change under irrelevant perturbations.

### Stage E: selective language realisation

The calibrated controller maps verified records to language. A factual sentence is always permitted if visually supported. An affective phrase is added only when it passes the calibrated decision rule.

## Experimental questions

1. Do current general and affect-specialised VLMs assign affect to the correct person in multi-person scenes?
2. Does a typed cue graph improve person--affect association and evidence localisation over free-form prompting?
3. Does counterfactual verification detect unsupported claims that confidence, entropy, or self-reported VLM certainty misses?
4. Does conformal selective reporting reduce affective claim error at useful coverage?
5. When face, body, and scene cues disagree, does distributional prediction better match annotator disagreement than a single label?
6. Does verified affect improve descriptions for readers, or do factual-only captions remain preferable in ambiguous cases?

## Baselines

- factual-only captioning;
- zero-shot and structured-prompt Qwen3-VL;
- Florence-2 plus a text generator;
- EmoVIT and Face-LLaVA as affect-specialised models;
- the thesis captioning pipeline as a transparent legacy baseline;
- SS-VLM without scene/body evidence;
- SEA-Cap, where code or reproducible outputs are available;
- confidence-only and entropy-only selective generation;
- no-intervention, random-region dropout, and attention-map-region verification.

## Evaluation

Report each failure axis separately.

### Affect and attribution

- person--affect association accuracy;
- multi-label macro-F1;
- Jensen--Shannon divergence or Earth Mover's Distance between predicted and human affect distributions;
- valence--arousal agreement;
- accuracy on the “insufficient evidence” class.

### Evidence quality

- cue-family macro-F1;
- pointing-game accuracy and region overlap where spatial annotations exist;
- necessity and insertion/deletion response;
- evidence-flip rate after cited-region removal;
- stability under non-affective perturbations;
- false-citation rate: a claim cites a region that has little measured influence.

### Selective reliability

- risk--coverage curves and area under the risk--coverage curve;
- empirical conformal coverage or controlled selective risk;
- Brier score and expected calibration error for the underlying affect distribution;
- error rates for direct, qualified, and abstained outputs.

### Language quality

- factual consistency using object-hallucination and image--text alignment measures;
- rate of unsupported affective predicates;
- naturalness and usefulness judged in a blinded human study;
- preference among factual-only, uncalibrated affective, and selectively affective captions.

### Stress tests

- number of people;
- face size, pose, and occlusion;
- face--context agreement versus conflict;
- image source and domain shift;
- demographic subgroup analysis where annotations and ethical conditions permit it.

## Essential ablations

- no person linking;
- no Action Units;
- no body expert;
- no scene/context expert;
- single categorical target instead of an affect distribution;
- no counterfactual verification;
- random masks instead of cited evidence masks;
- necessity only versus necessity plus stability/conflict;
- fixed confidence threshold versus conformal selection;
- free-form rationale versus typed evidence graph.

## Expected contribution statement

This work studies when a visually inferred affective statement is sufficiently supported to be communicated. It contributes (1) a person--cue--claim representation that separates observable evidence from affective interpretation, (2) an intervention-based verifier that tests whether cited visual evidence controls a candidate claim, and (3) a selective reporting protocol calibrated to state, qualify, or omit affective language. The resulting benchmark evaluates person attribution, affect disagreement, evidence faithfulness, and selective reliability separately, rather than reducing affect-aware reporting to caption similarity or emotion-label accuracy.

## WACV positioning and feasibility

The cleanest venue category is the **WACV Evaluation & Datasets Track**, which explicitly welcomes human-centred evaluation, audits, benchmarks, negative results, and tools that improve how evaluative claims are interpreted. The Algorithms Track is appropriate only if the graph fusion and risk controller are implemented and outperform strong alternatives; otherwise, the evaluation contribution is more defensible.

As of 14 August 2026, WACV 2027 Round 2 paper registration is due **21 August 2026**, the paper is due **28 August 2026**, and supplementary material is due **30 August 2026**. A new 1,000-image human-annotated benchmark and complete model cannot be produced credibly in two weeks. A WACV 2027 submission is realistic only as a compact audit paper if the images, model inference, and a small existing annotation effort are already available. The full AffectClaim paper should otherwise target the next appropriate WACV cycle or another later computer-vision venue.

For WACV 2027, the narrower emergency scope would be:

> **Do Vision--Language Models Know When Visual Affect Is Unsupported? A Counterfactual Audit**

This version would freeze all models, use 300--500 carefully stratified existing images, test several VLMs, and contribute the claim schema plus counterfactual/selective evaluation. It should not claim a new trained architecture.

## Twelve-week execution plan for the full paper

### Weeks 1--2: scope and pilot

- freeze the affect claim schema and ethical wording;
- sample 100 pilot images across the five strata;
- test annotation instructions and inter-annotator disagreement;
- reproduce baseline outputs from two open VLMs and the thesis models.

### Weeks 3--5: benchmark construction

- expand to 1,000--1,500 images;
- collect repeated annotations;
- audit face--body links, evidence regions, and “insufficient evidence” cases;
- define calibration, validation, in-domain test, and cross-domain test splits.

### Weeks 4--7: model and intervention pipeline

- build region/person linking and typed cue extraction;
- implement the person--cue--claim graph;
- add region deletion, restricted-view, cue-conflict, and nuisance tests;
- cache all specialist and VLM outputs for reproducibility.

### Weeks 7--9: calibration and baselines

- fit the structured fusion model;
- calibrate state/qualify/abstain decisions;
- compare confidence, entropy, random masks, attention masks, and counterfactual evidence scores.

### Weeks 9--10: evaluation

- run attribution, distribution, evidence, calibration, and language metrics;
- complete a blinded human comparison;
- conduct subgroup and failure-case analysis.

### Weeks 11--12: paper and release

- write the eight-page paper around one central result: error reduction as unsupported claims are selectively withheld;
- prepare annotation protocol, model cards, limitations, and data statement;
- release code, prompts, cached predictions, and the benchmark where licences permit.

## Go/no-go criteria

Proceed with the full claim only if the pilot shows all three:

1. Counterfactual evidence scores detect unsupported affective claims better than model confidence or entropy.
2. Person--cue linking reduces wrong-person affect attribution in multi-person scenes.
3. Selective reporting produces a clear risk reduction at non-trivial coverage, rather than achieving reliability only by abstaining almost everywhere.

If criterion 1 fails, the intervention is not yet a useful verifier. If criterion 2 fails, simplify the paper to a general affect-evidence audit. If criterion 3 fails, report the negative result honestly and examine which ambiguity strata make selective guarantees ineffective.

## Closest current work to position against

- Cai et al., **Towards Faithful Sentimental Image Captioning via Evidence-Aware Multi-Agent Reasoning (SEA-Cap)**, arXiv:2607.25789.
- Chen et al., **MultiEmo-Bench: Multi-label Visual Emotion Analysis for Multi-modal Large Language Models**, arXiv:2605.14635.
- Lin et al., **Why We Feel: Breaking Boundaries in Emotional Reasoning with Multimodal Large Language Models**, CVPR Workshops 2025.
- Kotte, **EVICT: Evidence-Sufficiency Verification via Counterfactual Dropout for Visually-Grounded Selective Question Answering**, CVPR Workshops 2026.
- Chaubey et al., **Face-LLaVA: Facial Expression and Attribute Understanding through Instruction Tuning**, WACV 2026.
- Xie et al., **EmoVIT: Revolutionizing Emotion Insights with Visual Instruction Tuning**, CVPR 2024.
- Kosti et al., **EMOTIC: Emotions in Context Dataset**, CVPR Workshops 2017.

The novelty should be described as the integration of person-specific affect claims, typed multimodal evidence, intervention-based claim verification, observer disagreement, and calibrated language selection. It should not be described as the first use of grounding, counterfactual dropout, conformal prediction, or VLM-based emotion reasoning individually.

