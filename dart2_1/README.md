# DART-2.1 — Invariant Rule Factorization

DART-2.1 tests whether a **single frozen rule graph** can explain multiple source tasks with only small explicit task parameters. It strengthens DART-2.0 with strict controls for theta semantics and shared-rule validity.

## Core hypothesis

Find one shared rule graph `G*` such that:

```text
G*(theta_add)      -> add
G*(theta_mul)      -> mul
G*(theta_compose)  -> compose
```

Then freeze `G*` and fit only `theta_sub` for the unseen related task.

## Research controls

- full task-to-task theta permutation matrix
- random graph control with matched structure/parameter budget
- separately trained graph per source task
- rule-level causal intervention fidelity
- shared-vs-separate graph parity
- matched MLP control
- frozen-graph theta-only transfer

## Key interpretation

A low loss alone is not sufficient. A candidate should ideally have meaningful source capability, stable task parameters, non-trivial parameter specificity, rule-level causal fidelity, and performance close to separately trained graphs while using a single shared graph.

## Full run

```bash
python3 dart021.py \
  --seeds 1 2 \
  --all-tasks add compose mul sub \
  --holdout-tasks sub \
  --contrast-tasks sort \
  --teacher-steps 800 \
  --core-fit-steps 300 \
  --theta-fit-steps 120 \
  --target-theta-fit-steps 400 \
  --transfer-control-steps 400 \
  --separate-control-steps 200 \
  --train-size 6000 \
  --verifier-size 1500 \
  --test-size 1500 \
  --rel-samples-per-task 2048 \
  --fit-batch-samples 512 \
  --device cuda
```

The resulting `dart021_results.json` is the source of truth for the experiment.
