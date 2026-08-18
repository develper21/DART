# DART-2.4 Theory Note

## Motivation

DART-2.3 found a strong learned-placement signal but weak direct causal fidelity. That leaves an ambiguity: an adapter location can matter for accuracy without being the location that causally carries task-specific algorithmic variation.

## Hypothesis

Let the shared skeleton be `G` and task-specific adapter sites be `A_t,i`. If slot `i` is causally responsible for task variation, ablating that slot from both task-specific rules should substantially reduce the difference between the two task behaviors.

For a task pair `(a,b)` and slot `i`:

```text
D_full  = || F_a(x) - F_b(x) ||
D_abl   = || F_a^{-i}(x) - F_b^{-i}(x) ||
slot_causal_necessity_i = max(0, 1 - D_abl / (D_full + eps))
```

This is a causal-necessity diagnostic over task differences, not a generic feature-importance score.

## Controls

DART-2.4 keeps the previous falsification controls:

- random placement with the same number of active adapters
- separate per-task graph control
- operator permutation matrix
- MLP control on holdout tasks
- adapter-effect and stability measurements

## Acceptance logic

A candidate requires:

- source capability above the source floor
- stable task adapters
- operator specificity above threshold
- random-placement advantage
- shared-vs-separate parity within the configured bound
- adapter effect above threshold
- slot causal necessity above threshold

The legacy rule-causal score is retained for diagnosis but is not the sole causal gate.

## Research question

The key question is:

> Can DART identify sparse internal sites that are both useful for task variation and causally necessary for that variation?

If yes, the next step is frozen-skeleton transfer using only the causally validated slots. If no, the failure tells us that internal slot locality is not the correct abstraction.
