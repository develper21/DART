# DART-1.8

## Causal Bottleneck / Minimum Computational Subgraph Discovery

DART-1.7 introduced a typed intermediate representation `E -> T(theta) -> D`, but better interface invariance did not translate into strong unseen-task transfer. DART-1.8 tests a deeper hypothesis: the useful computation may occupy only a small subset of the intermediate representation, and that subset should be **causally necessary**, not merely correlated.

### Core pipeline

```text
hidden state x
   |
   v
Extractor E
   |
   v
full relation R
   |
   +--> causal importance estimator
   |
   v
minimal bottleneck Z
   |
   v
T(Z, theta)
   |
   v
Decoder D
   |
   v
replacement output
```

### Controls

- Full typed representation is retained as the conceptual DART-1.7 baseline.
- Learned causal bottleneck.
- Random same-width bottleneck control.
- Matched MLP control.

### Hard gates

A candidate must satisfy source capability, theta necessity/stability, interface invariance, bottleneck sufficiency, causal necessity of selected features, and contrast-task specificity. Cheapness alone cannot make a candidate eligible.

### Main scientific question

> Can DART identify the minimum causally necessary intermediate information for the computation, and can that frozen bottleneck transfer to an unseen related task after fitting only tiny task parameters?

### Status

Implementation + smoke test only until the full CUDA experiment is completed. Smoke outputs are not research results.
