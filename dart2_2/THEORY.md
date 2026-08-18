# DART-2.2 Theory

## Motivation

DART-2.0 showed that structured rule graphs can have non-trivial causal fidelity and outperform matched random graphs. DART-2.1 then tested whether the same shared graph could explain several source tasks nearly as well as separate graphs. That sharing hypothesis looked promising, but the scalar task parameterization was not task-specific: the full permutation matrix showed almost no advantage for the correct parameter assignment.

## Hypothesis

A reusable computation can be factorized into:

```text
G* + O_task
```

where `G*` is a frozen shared computational mechanism and `O_task` is a tiny structured operator that expresses task-specific algorithmic variation.

## Falsification requirements

A candidate should not be called a reusable rule unless:

- source capability is meaningful,
- `O_task` has measurable computational effect,
- correct operator assignment beats wrong assignments,
- rule interventions agree with teacher interventions,
- the learned shared graph beats a matched random graph,
- shared-graph capability is close to separately trained graphs,
- and `G*` remains frozen during target transfer.

Failure is diagnosed as graph capacity, operator specificity, causal fidelity, or transfer failure rather than collapsed into a single score.
