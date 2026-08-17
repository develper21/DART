# DART-1.4

**Dynamic Algorithm Replacement Training — Factorized Invariant Primitive Compilation**

DART-1.4 asks a narrower question than earlier versions:

> Can several tasks share the same computational mechanism while differing only by a tiny explicit operation parameter?

## What changed from DART-1.3

DART-1.3 used one standalone structured primitive and tested frozen transfer. DART-1.4 factorizes that primitive into a small shared basis and a tiny coefficient vector `theta`.

```text
DART-1.3
source tasks → one structured primitive → frozen transfer

DART-1.4
source tasks → shared basis B1...Bm + tiny theta_T
                                ↓
                         freeze shared basis
                                ↓
                    fit only theta_holdout
```

There is **no task ID, neural task embedding, learned conditioner, or target residual network** in the DART primitive.

## Experimental protocol

### Meta tasks

Default source tasks:

- `add`
- `compose`
- `mul`

### Related holdout

- `sub`

### Contrast holdout

- `sort`

The contrast task is intentionally outside the main arithmetic family so that transfer specificity can be tested.

## Candidate families

Default structured families are:

- affine polynomial
- low rank
- polynomial
- diagonal

The MLP is retained only as a neural control and is not eligible to win the DART search.

## Evaluation

DART-1.4 reports:

- meta-task accuracy before/after theta adaptation;
- relational agreement;
- causal/directional agreement;
- source-theta stability across split observations;
- shared-core parameter/MAC counts;
- related-task zero-shot accuracy;
- related-task accuracy after fitting only `theta`;
- matched MLP control;
- contrast-task behavior.

## Full experiment

```bash
python3 dart014.py \
  --seeds 1 2 \
  --all-tasks add compose mul sub \
  --holdout-tasks sub \
  --contrast-tasks sort \
  --teacher-steps 800 \
  --core-fit-steps 300 \
  --theta-fit-steps 100 \
  --meta-theta-adapt-steps 200 \
  --target-theta-fit-steps 400 \
  --surgery-rounds 2 \
  --transfer-control-steps 400 \
  --train-size 6000 \
  --verifier-size 1500 \
  --test-size 1500 \
  --rel-samples-per-task 2048 \
  --rel-directions 4 \
  --theta-dim 4 \
  --intervention-eps 0.05 \
  --directional-weight 0.50 \
  --relational-weight 0.50 \
  --theta-stability-weight 0.25 \
  --device cuda
```

## Interpretation

The strongest result is not simply high holdout accuracy. We are looking for:

```text
same shared basis
+
small stable theta
+
near-teacher related-task capability
+
no task-specific neural machinery
+
clear degradation on unrelated tasks
```

That combination would be stronger evidence of a reusable computational primitive than ordinary compression or distillation.

## Status

**Implementation:** complete and smoke-tested.

**Full CUDA research validation:** pending.

DART-1.4 is an experimental research hypothesis, not a claim that reusable algorithm discovery has already been solved.
