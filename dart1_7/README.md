# DART-1.7

**Dynamic Algorithm Replacement Training — Typed Intermediate Representation Discovery**

DART-1.7 asks whether reusable computation is easier to discover after mapping hidden states into a structured intermediate representation.

## Core idea

```text
hidden state x
   │
   ▼
structured extractor E
   │
   ▼
typed relation representation R
   │
   ▼
structured transform T(R, θ)
   │
   ▼
structured decoder D
   │
   ▼
replacement output
```

No learned task embedding, large residual, or MLP is part of the DART primitive. A small explicit `theta` remains and is tested for necessity and causal effect.

## What changed from DART-1.6
DART-1.6 composed structured operators directly on the hidden vector. DART-1.7 introduces a typed intermediate representation: projected values, centered values, squares, and adjacent-difference relations. This is a research hypothesis, not a claim that these relations are the final algorithmic representation.

## Candidate families
- diagonal
- polynomial
- affine_polynomial
- low_rank

MLP is used only as a control.

## Evaluation
The experiment checks:

1. Source-task capability and worst-task capability.
2. Theta effect and theta stability.
3. Cross-task interface invariance.
4. Frozen zero-shot transfer to related `sub`.
5. Theta-only transfer adaptation.
6. Contrast behavior on `sort`.
7. Matched MLP control.

## Run

```bash
python3 dart017.py \
  --seeds 1 2 \
  --all-tasks add compose mul sub \
  --holdout-tasks sub \
  --contrast-tasks sort \
  --teacher-steps 800 \
  --core-fit-steps 300 \
  --theta-fit-steps 120 \
  --target-theta-fit-steps 400 \
  --transfer-control-steps 400 \
  --train-size 6000 \
  --verifier-size 1500 \
  --test-size 1500 \
  --rel-samples-per-task 2048 \
  --typed-dim 8 \
  --theta-delta 0.25 \
  --min-avg-source-acc 0.30 \
  --min-worst-source-acc 0.22 \
  --min-theta-effect 0.02 \
  --max-theta-stability 0.75 \
  --min-interface-invariance 0.60 \
  --device cuda
```

The output is `dart017_results.json`.

## Interpretation
A positive result requires more than compression. The strongest outcome would be a stable structured representation and shared transform that transfers to `sub` with only tiny theta adaptation, while `sort` remains poorly explained and the structured primitive competes with the neural control at much lower compute.
