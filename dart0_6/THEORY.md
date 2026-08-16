# DART-0.6 Theory — Trajectory Compression

## Motivation

DART-0.1 through DART-0.5 repeatedly replaced a **single FFN**. Even after
switching from hidden-state imitation to downstream behavioural optimization,
DART-0.5 selected `small_mlp` in all 8 surgery rounds. The research backup
therefore identifies the replacement unit as the likely bottleneck.

## New hypothesis

The useful computation may be distributed across several Transformer blocks:

`h0 -> h1 -> h2 -> h3`

Rather than searching for an independent replacement of one block, DART-0.6
searches for a **reusable transition operator O**:

`O(h0) ≈ h1`, `O(h1) ≈ h2`, `O(h2) ≈ h3`

The same operator is then applied three times after surgery.

## Why this is different from DART-0.5

DART-0.5 asks whether one local sub-computation can be replaced cheaply.
DART-0.6 asks whether a repeated transformation across a multi-layer
trajectory can be identified, compiled, and transferred.

The reusable object is therefore the **transition motif**, not a single hidden
subgraph.

## Candidate families

- identity
- shared diagonal affine
- shared polynomial
- shared low-rank residual
- shared primitive mixture (`tanh`, `abs`, square)
- shared MLP control

The shared MLP is retained as a control to test whether even a trajectory-level
search simply collapses back to neural approximation.

## Acceptance evidence

Strong evidence would require all of:

1. a compact non-MLP transition wins at least some rounds;
2. its trajectory consistency remains high on held-out trajectories;
3. post-surgery adaptation retains/improves downstream capability;
4. the same transition provides positive transfer on a related task;
5. the effect survives multiple seeds and repeated surgery.

No novelty claim is made from a single positive run.
