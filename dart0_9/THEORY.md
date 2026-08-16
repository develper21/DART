# DART-0.9 — Routing-Preserving Structured Replacement

## Research question

DART-0.8 showed that preserving the Transformer's attention/information-routing path largely restores capability after aggressive FF replacement. The remaining problem is that an MLP still wins every replacement search.

DART-0.9 therefore freezes the successful routing-preserving scaffold and asks a narrower question:

> Can a structured, non-MLP computational primitive replace the FF computation while matching a neural control under the same routing pathway and compute constraint?

## Main hypothesis

For each retained Transformer block:

`u = x + Attention(Norm1(x))`

`y = u + C(Norm2(u)) + R_i(Norm2(u))`

where `C` is shared across the replacement span and `R_i` is a small block-specific residual adapter.

## Candidate policy

### DART-eligible structured candidates

- identity
- diagonal affine
- polynomial
- affine-polynomial composition
- low-rank linear

The MLP is intentionally excluded from DART selection.

### Neural controls

1. `mlp_control`: same routing-preserving interface, trained directly on task labels.
2. `distilled_mlp_control`: same interface, trained against teacher logits plus task loss.

These controls are reported but can never become the DART winner.

## Required evidence

A convincing DART mechanism needs more than compression:

1. capability retention close to the teacher;
2. lower replacement compute;
3. routing agreement;
4. structured candidate competitive with or better than the neural controls;
5. positive replacement advantage against the best control;
6. cross-task transfer that is at least as good as the matched control.

## Falsification

DART-0.9 should be considered negative if structured candidates consistently trail the neural controls after compute matching, even when routing is preserved.

That would mean routing preservation is sufficient for compression, but not sufficient for algorithmic replacement.
