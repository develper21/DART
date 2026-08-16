# DART-1.0 Theory

## Why 1.0 exists

DART-0.8 established that preserving attention/information routing is important for capability retention. DART-0.9 then restricted DART selection to structured operators, successfully preventing the MLP from becoming the DART winner, but transfer to another task remained negative.

The unresolved question is therefore not simply whether a structured operator can replace an FFN. It is whether the operator is **reusable across tasks**.

## New hypothesis

Let tasks `T1, T2, T3` provide trajectories through their routing-preserving Transformer blocks. We search for one shared structured operator `C*` that minimizes joint downstream loss across all meta-training tasks:

`C* = argmin_C mean_i L(T_i, C, R_i) + complexity(C)`

where `R_i` are small task-specific residual adapters and the attention routers remain frozen.

After discovery, freeze `C*`.

For held-out task `T4`:

`T4 -> routing-preserving scaffold + frozen C* + tiny R4`

Only `R4` may adapt.

## Core scientific test

The primitive should be useful **before** target-task adaptation. Residual-only adaptation is a second measurement, not the primary proof.

Therefore report:

- zero-shot held-out accuracy;
- residual-adapted held-out accuracy;
- matched neural control;
- shared-core compute and parameter count;
- routing agreement;
- transfer gain over scratch/control.

## Falsification

DART-1.0 is negative if the frozen shared structured primitive has no meaningful held-out capability or if its adapted performance is consistently explained by the target residual rather than the shared primitive.
