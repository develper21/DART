# DART-2.2 — Structured Task-Operator Discovery

DART-2.2 is the next step after DART-2.1. DART-2.1 provided evidence that one shared rule graph can reach nearly the same source-task capability as separate per-task graphs, but its scalar task parameters had almost no task specificity. DART-2.2 therefore keeps the shared graph and replaces weak scalar configuration with a small, explicit **task-operator grammar**.

## Research question

> Can one frozen computational rule graph `G*` implement multiple algorithms when the task-specific difference is expressed by a compact structured operator `O_task`, and can `O_sub` be discovered for an unseen related task without changing `G*`?

## Core decomposition

```text
input
  ↓
shared rule graph G*
  ↓
structured task operator O_task
  ↓
output
```

The task operator is selected from a constrained grammar:

- identity
- scale
- negate
- difference
- product
- mix

There is no task embedding, large conditioner, target-task residual, or MLP inside the DART candidate.

## Validation protocol

DART-2.2 evaluates:

1. Source capability and worst-source capability.
2. Operator stability across data splits.
3. Operator effect: whether the operator actually changes computation.
4. Full operator-permutation matrix: correct `O_task` versus wrong operators.
5. Rule-level causal fidelity under node intervention.
6. Random-graph control at matched structure.
7. Shared-graph versus separately trained task graphs.
8. Frozen shared graph with target-task operator-only adaptation.
9. Matched MLP control.
10. Related `sub` holdout and unrelated `sort` contrast.

## Reproducibility

```bash
python3 dart022.py \
  --seeds 1 2 \
  --all-tasks add compose mul sub \
  --holdout-tasks sub \
  --contrast-tasks sort \
  --teacher-steps 800 \
  --core-fit-steps 300 \
  --operator-fit-steps 120 \
  --target-operator-fit-steps 400 \
  --transfer-control-steps 400 \
  --separate-control-steps 200 \
  --train-size 6000 \
  --verifier-size 1500 \
  --test-size 1500 \
  --rel-samples-per-task 2048 \
  --fit-batch-samples 512 \
  --device cuda
```

The smoke test is only an implementation check; it is not a research result.
