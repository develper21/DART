# DART-1.4 Theory

## Core hypothesis

DART-1.3 removed task-specific learned conditioning and showed that a standalone structured primitive can have task-family-specific behavior, but transfer to the related holdout remained weak. DART-1.4 tests a stronger factorization:

**one shared computational mechanism `C` + a tiny explicit operation parameter `theta` → multiple task behaviors.**

The important distinction is that `theta` is **not** a learned task embedding and is not implemented as a neural conditioner. It is a small vector of explicit coefficients used to combine a shared structured basis.

## Discovery

For source tasks `T1 ... Tk`, DART jointly learns:

```text
shared primitive basis B1(x), ..., Bm(x)
```

and per-task explicit coefficients:

```text
theta_T1, ..., theta_Tk
```

such that:

```text
y_T(x) ≈ Σ_j theta_T[j] * Bj(x)
```

The same basis must explain all meta-tasks. A candidate is additionally evaluated on relational / directional behavior and on theta stability across split observations.

## Transfer

After discovery, the shared basis is frozen.

For an unseen related task, only `theta_holdout` is fitted from a small adaptation set. No task embedding, conditioner, or target residual network is trained.

Two baselines are reported:

1. Zero-shot using the centroid of source-task theta vectors.
2. Matched routing-preserving MLP control trained on the same target data budget.

## Research question

Does a common computational mechanism survive across tasks when only a tiny explicit operation parameter changes?

A strong DART-1.4 result would look like:

```text
source task A ─┐
source task B ─┼─> same B_j(x)
source task C ─┘      + tiny theta_T
                         ↓
                    freeze basis
                         ↓
                    unseen task D
                         ↓
               fit only theta_D
                         ↓
                 useful capability
```

An unrelated contrast task should remain substantially worse than the related holdout if the primitive is genuinely family-specific.

## Falsification criteria

DART-1.4 is not considered successful merely because it compresses the FFN. Evidence against the hypothesis includes:

- holdout adaptation requires large or unstable theta;
- theta changes do not improve beyond the centroid zero-shot solution;
- the factorized primitive is consistently worse than the matched MLP control;
- unrelated tasks behave similarly to related tasks;
- a different basis is required independently for every task.
