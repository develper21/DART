# DART-2.6 Theory

## From reconstruction to mediation

DART-2.5 showed that a slot can sometimes reconstruct teacher-like intervention behavior without proving causal necessity. DART-2.6 therefore treats reconstruction as only one part of a stronger causal test.

For a candidate mediator `Z`, we seek:

`remove(Z) -> behavior changes`

and

`restore / strengthen(Z) -> teacher-aligned behavior returns`

while varying intervention strength `alpha`.

## Causal Mediation Efficiency

The candidate is scored using a multiplicative objective over:

- necessity,
- sufficiency,
- teacher-aligned trajectory fidelity,
- minimality,

with an active-slot complexity penalty.

This makes a high reconstruction score alone insufficient.

## Controls

DART-2.6 retains:

- shared-vs-separate graph control,
- random placement control,
- operator permutation control,
- MLP control,
- frozen shared skeleton for target-task adaptation.

## Transfer

Source tasks are used for discovery. The shared skeleton is frozen and only the target-task adapter configuration is adapted on the holdout task.
