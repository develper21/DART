# DART-2.0 Changelog

## DART-2.0

### Added

- Computational rule-graph discovery as the primary abstraction.
- Sequential, parallel-sum, and residual-parallel motifs.
- Structured rule nodes: diagonal, polynomial, affine-polynomial, and low-rank.
- Joint source-task fitting of one shared graph with task-specific theta.
- Rule-level causal intervention diagnostics.
- Theta-permutation control.
- Same-budget random-graph control.
- Frozen graph + theta-only holdout transfer.
- Related-task and contrast-task evaluation.
- Explicit failure-oriented diagnostics through gate eligibility.

### Design shift

DART-1.x progressively investigated state, representation, bottleneck, and state-interchangeability hypotheses. DART-2.0 makes the **computational transformation rule itself** the primary object of discovery.

### Validation

A tiny CPU smoke test was run during implementation to validate:

- teacher training,
- graph fitting,
- theta fitting,
- rule intervention,
- permutation control,
- random graph comparison,
- frozen transfer,
- MLP baseline,
- result serialization.

The smoke test is not a research result.
