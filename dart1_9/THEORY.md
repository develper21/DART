# DART-1.9 Theory

## Hypothesis

A reusable computational primitive should admit a **causally meaningful, cross-task interchangeable internal state**.

DART-1.8 showed that a compact bottleneck can be information-sufficient without being causally necessary or task-specific. DART-1.9 therefore replaces feature-level sufficiency with a direct intervention criterion.

## Counterfactual criterion

Let `Z_A` be a structured state extracted from task A and `Z_B` the corresponding state from task B. Insert `Z_B` into the frozen task-A downstream computation while holding the rest of the mechanism fixed.

A useful transferable variable should produce:

```text
learned swap effect >> random swap effect
```

and the direction/magnitude of the effect should agree with the teacher-grounded downstream response.

## Acceptance gates

- source accuracy above minimum;
- worst source task above minimum;
- theta has a measurable effect;
- theta is stable across splits;
- learned counterfactual swap beats the randomized control by a margin;
- held-out related task is not merely solved by unconstrained retraining.

## Failure diagnosis

- **T fail:** theta is inactive or unstable.
- **Causal-swap fail:** the representation is predictive but not portable under intervention.
- **Transfer fail:** causal interchangeability exists locally but does not generalize to the held-out task.
- **Control fail:** the apparent gain is not better than an ordinary neural replacement.

## What would count as a meaningful positive result

The strongest outcome is:

```text
same structured mechanism
+ task-specific state/configuration
+ frozen mechanism at transfer
→ near-teacher behavior on a related unseen task
```

while random swaps and an unrelated task remain substantially worse.
