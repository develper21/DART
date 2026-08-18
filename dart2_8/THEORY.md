# DART-2.8 Theory

## Hypothesis

DART-2.7 provided evidence that a small causal component can be localized and shared across source tasks. DART-2.8 tests the next claim: the same causal primitive can be frozen, while only a very small explicit task transformation changes its behavior enough to transfer to an unseen related task.

## Frozen object

`G*` and the discovered primitive base state are frozen after source discovery.

## Task variation

Each active primitive slot receives only two trainable scalars:

- `phi_scale`
- `phi_shift`

with

`raw_effective = raw_base * (1 + phi_scale) + phi_shift`.

This is deliberately much smaller than fitting the full adapter parameterization.

## Controls

- zero-shot frozen primitive
- tiny two-scalar reparameterization
- full adapter reparameterization on the same frozen skeleton
- primitive permutation control
- random primitive control
- matched MLP control
- contrast-task transfer

## Success condition

The strongest outcome is a small or near-teacher transfer gap with a frozen primitive, a small `phi` parameter count, and strong rejection of the wrong-primitive and random-primitive controls.

A failure in tiny transfer while full frozen-skeleton fitting succeeds indicates that the primitive is reusable in principle but the chosen low-dimensional reparameterization is too restrictive.
