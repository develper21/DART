# DART-1.9

## Counterfactual Causal Interchangeability Discovery

DART-1.9 is the next experimental step in the DART research program. It builds on the tested DART-1.7 typed intermediate-representation pipeline and DART-1.8 bottleneck work, but changes the central question from **"is this representation predictive?"** to **"is this representation causally portable between related tasks?"**

### Core hypothesis

A reusable computational variable should not merely preserve information or correlate with the output. If the variable is genuinely reusable, replacing the state produced by one related task with the corresponding state from another task should cause a predictable change in the frozen downstream computation.

DART-1.9 therefore measures **counterfactual interchangeability** directly.

```text
Task A teacher state
        |
        v
   extract relation
        |
        v
       Z_A -------------------+
                               |
                         counterfactual swap
                               |
       Z_B -------------------+
        |
        v
  same frozen mechanism
        |
        v
 predicted/actual output change
```

### What DART-1.9 adds

- Teacher-grounded counterfactual state swaps across related source tasks.
- A **random-swap control** with the same intervention budget.
- Swap-fidelity and swap-margin metrics.
- Explicit hard eligibility gate: a candidate must outperform the random swap control by a minimum margin.
- Frozen-primitve transfer to a related held-out task (`sub`) and a contrast task (`sort`).
- DART-1.7 style structured E→T(θ)→D primitive and theta necessity/stability checks.

### Research question

> Can a structured computational state discovered on one task be causally substituted with the corresponding state from another related task, while preserving a predictable change in the frozen network's behavior?

If the answer becomes yes at meaningful capability, this is stronger evidence for a reusable computational abstraction than representation similarity or feature ablation alone.

### Controls

1. Structured DART candidate.
2. Random counterfactual swap control.
3. Matched MLP control for final transfer evaluation.

### Full CUDA run

```bash
python3 dart019.py \
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
  --rel-dim 12 \
  --theta-delta 0.25 \
  --min-avg-source-acc 0.30 \
  --min-worst-source-acc 0.22 \
  --min-theta-effect 0.02 \
  --max-theta-stability 0.75 \
  --min-swap-margin 0.05 \
  --device cuda
```

### Important metrics

- `swap_fidelity`: teacher-grounded fidelity of a learned cross-task state swap.
- `random_swap_fidelity`: same measurement for a randomized control.
- `swap_margin = swap_fidelity - random_swap_fidelity`.
- `dart_zero` / `dart_adapted`: held-out transfer before and after tiny theta adaptation.
- `vs_mlp_adapt`: DART relative to the neural control.

A promising result requires **learned swap fidelity to clearly beat random swap fidelity**, while source capability remains meaningful and the same mechanism transfers to the related holdout.

### Scientific caution

DART-1.9 is an experiment, not a claim of general algorithm discovery. Positive results on the synthetic arithmetic benchmark would be evidence for the mechanism, not proof of generality to large language models.
