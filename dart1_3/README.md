# DART-1.3

**Causal/Relational Primitive Compilation**

DART-1.3 is the next experiment after DART-1.2. It addresses the remaining bottleneck: a behavioral signature can describe a task, but a learned conditioner did not make the discovered primitive transferable.

## Core idea

Instead of learning a task-conditioned replacement, DART-1.3 extracts local computational invariants from the teacher's internal FF computation:

```text
value response
+ directional response
+ interaction response
        ↓
shared structured primitive
        ↓
freeze
        ↓
related holdout / contrast holdout
```

No task ID, learned task code, or large target-task residual network is used by the DART primitive.

## Candidate families

- identity
- diagonal
- polynomial
- affine_polynomial
- low_rank

An MLP is retained only as a neural control and is never eligible as the DART winner.

## Main metrics

- downstream capability
- routing agreement
- attention ablation drop
- relational agreement
- causal intervention agreement
- parameter count
- MAC estimate
- transfer gain versus the teacher
- transfer gap versus the MLP control
- related-vs-contrast specificity

## Full run

```bash
python3 dart013.py \
  --seeds 1 2 \
  --all-tasks add compose mul sub \
  --holdout-tasks sub \
  --contrast-tasks sort \
  --teacher-steps 800 \
  --core-fit-steps 300 \
  --adaptation-steps-per-round 400 \
  --surgery-rounds 2 \
  --train-size 6000 \
  --verifier-size 1500 \
  --test-size 1500 \
  --rel-samples-per-task 2048 \
  --rel-directions 4 \
  --intervention-eps 0.05 \
  --relational-weight 0.50 \
  --causal-weight 0.50 \
  --directional-weight 0.50 \
  --interaction-weight 0.25 \
  --device cuda
```

## Interpretation

A strong DART-1.3 result is not merely high accuracy. The target is a frozen structured primitive that performs well on a related unseen task, clearly underperforms on a contrast task, and remains competitive with the neural control at much lower computation.
