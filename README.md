# DART-1.0 — Joint Shared Primitive Discovery + Frozen Unseen-Task Transfer

## Research question

DART-0.9 showed that structured replacement can work inside a routing-preserving Transformer, but cross-task transfer remained negative.

DART-1.0 changes the unit of discovery:

> **Discover one structured computational primitive jointly across multiple meta-training tasks, freeze that primitive, then insert it into an unseen related task.**

The core is shared across every meta-task model and across every replaced block in those models. Each meta-task keeps a small task-specific residual adapter. During held-out transfer, the shared core is frozen; only the target residual adapter may adapt.

## Important controls

- The DART search is restricted to structured candidates: identity, diagonal, polynomial, affine-polynomial, low-rank.
- A matched routing-preserving MLP control is trained on the held-out task but is never eligible to become the DART primitive.
- The held-out evaluation reports zero-shot shared-core performance and residual-only adaptation performance.

## What counts as success

The strongest evidence would be:

1. a structured primitive wins jointly across multiple meta-training tasks;
2. the same frozen primitive retains capability on a held-out task without retraining the primitive;
3. small residual-only adaptation recovers further capability;
4. transfer is competitive with or better than a matched MLP control;
5. routing agreement remains high;
6. the primitive is substantially cheaper than the original FF computation.

## Default experiment

Meta-training tasks are all tasks except the holdout. Default holdout is `sub`, so the primitive is discovered jointly from `add + compose + mul` and then frozen for `sub`.

```bash
python3 dart010.py \
  --seeds 1 2 \
  --all-tasks add compose mul sub \
  --holdout-tasks sub \
  --teacher-steps 800 \
  --core-fit-steps 300 \
  --adaptation-steps-per-round 400 \
  --surgery-rounds 2 \
  --transfer-adaptation-steps 400 \
  --train-size 6000 \
  --verifier-size 1500 \
  --test-size 1500 \
  --residual-rank 2 \
  --residual-weight 0.01 \
  --routing-weight 0.20 \
  --ablation-weight 0.10 \
  --device cuda
```

The JSON records the shared-core parameters, zero-shot held-out accuracy, residual-adapted accuracy, matched MLP control, and the transfer gains.
# DART
