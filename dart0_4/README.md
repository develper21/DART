# DART-0.4

## What this release changes

DART-0.3 revealed that our old DART implementation was effectively
"small-FFN distillation + surgery + adaptation." That is not enough.

DART-0.4 changes operator discovery:

    observe internal computation
        -> create controlled interventions
        -> fit multiple operator families
        -> compare value response + directional response
        -> prefer the simplest/cheapest adequate operator
        -> surgically replace
        -> adapt
        -> repeat

Candidate families:
- diagonal affine
- polynomial
- low-rank
- full linear
- small MLP

The replacement is tested on perturbed hidden states rather than only
the original trace points.

## Run on your GPU

Recommended staged run:

```bash
python3 dart04.py \
  --seeds 1 2 \
  --tasks add compose \
  --teacher-steps 800 \
  --adaptation-steps-per-round 400 \
  --surgery-rounds 2 \
  --operator-fit-steps 250 \
  --transfer-adaptation-steps 400 \
  --train-size 6000 \
  --test-size 1500 \
  --trace-rows 10000 \
  --latency-iters 30
```

Full/default run:

```bash
python3 dart04.py
```

## How to interpret

The important information is no longer just final accuracy.

Inspect:
1. Which operator family wins in each round?
2. Does the selected operator remain cheap?
3. Does adaptation restore capability after surgery?
4. Does round 2 find another useful replacement?
5. Does a source-task operator transfer to a structurally related target task?
6. Is CUDA latency positive and reproducible?

A promising result would show non-MLP operators repeatedly selected,
successful post-surgery adaptation, and transfer benefits.

If the small MLP wins every round and nothing transfers, DART-0.4 has
not yet escaped ordinary distillation/compression.
