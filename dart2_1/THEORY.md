# DART-2.1 Theory Note

DART-2.0 showed that structured rule graphs can obtain non-trivial rule-causal fidelity and beat same-budget random graphs, but the shared rule did not yet transfer strongly to an unseen task.

DART-2.1 asks a stricter question: **is the discovered rule graph itself invariant across tasks, with task variation confined to a small explicit parameter vector?**

The core decomposition is:

```text
shared invariant graph G*
        +
small task configuration theta_task
        ↓
algorithmic behavior
```

Three controls make the claim harder to fake:

1. **Theta permutation matrix**: the correct task/theta pairing should outperform wrong pairings.
2. **Random graph control**: a fresh graph with the same structural family and parameter budget should perform worse.
3. **Separate graph parity**: one shared graph should approach the performance of separately trained per-task graphs; a large gap means the shared factorization is not yet adequate.

Rule-level intervention fidelity remains because transfer must reflect causal computation rather than merely predictive fit.

A successful DART-2.1 result requires the same `G*` to remain useful after freezing while only `theta_sub` is adapted on the unseen related task.
