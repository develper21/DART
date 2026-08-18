# DART-2.3

## Interleaved Shared–Task Factorization

DART-2.3 tests whether algorithmic variation is localized inside an otherwise shared computational skeleton.

### Core hypothesis

Instead of:

`shared G -> one task operator -> output`

DART-2.3 uses:

`G1 -> A_t,1 -> G2 -> A_t,2 -> G3 -> A_t,3`

where the `G` stages are shared structured primitives and only a small number of task adapters are active.

### Research controls

- random adapter placement with matched active-adapter count
- task-parameter permutation matrix
- separate per-task graph control
- rule-level causal intervention
- adapter effect / necessity
- frozen shared skeleton with target-task adapter fitting
- related `sub` holdout and unrelated `sort` contrast
- matched MLP control

### Research interpretation

A strong DART-2.3 result requires a shared skeleton close to separate graphs, meaningful active adapters, better learned placement than random placement, and useful frozen transfer to `sub`.
