# DART-0.7 — Shared Computational Core + Minimal Residual

## Research hypothesis

DART-0.6 showed that a single shared operator can fit repeated latent transitions with high trajectory consistency, but replacing the whole multi-layer span with that operator destroys task capability. DART-0.7 tests a narrower hypothesis:

> A trajectory may contain a reusable computational core that explains a substantial fraction of each transition, while small position-specific residuals preserve task-specific information.

The replacement is therefore:

`h_{i+1} = C(h_i) + R_i(h_i)`

where:

- `C` is one shared computational core reused at every trajectory step.
- `R_i` is a small, step-specific residual adapter.
- The residual is explicitly complexity-penalized.

## What would count as evidence?

A useful result requires all of the following to move in the desired direction:

1. Capability after surgery remains reasonably close to the teacher and clearly above DART-0.6.
2. The shared core explains a large fraction of the transition, so residuals remain small.
3. The selected core is often non-neural or lower-complexity than the neural control.
4. Adaptation improves rather than merely relearns the full task.
5. A discovered core transfers to a related target task better than a scratch control.

## What would falsify the idea?

- Residuals become effectively full-sized neural networks.
- The shared core contributes little and the residual performs the real computation.
- The neural control wins consistently with a much smaller residual burden.
- Transfer disappears or reverses after compute-matched evaluation.

## Important distinction from DART-0.6

DART-0.6 used:

`h0 -> O -> O -> O`

with no residual path.

DART-0.7 uses:

`h0 -> C+R1 -> C+R2 -> C+R3`

The core is still shared; only the correction is allowed to vary by step.
