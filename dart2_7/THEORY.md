# DART-2.7 Theory

## Hypothesis

A causal effect can be distributed across several interacting task-specific components. Therefore, searching only for a single causal slot can understate the true reusable computation.

## Factorization

`Shared skeleton + small task adapters` is retained. The research object is now a subset `S` of active slots.

## Intervention set analysis

For each non-empty subset `S`, the implementation measures the downstream logit change produced by ablating `S` and compares it with the teacher's corresponding feed-forward intervention direction.

## Minimal causal set

The minimal set is the smallest subset whose teacher-aligned intervention score reaches the configured fidelity threshold.

## Localization / redundancy / synergy

Let `J` denote the full active-set score and `B` the best singleton score. Causal concentration is `B/J`. A high concentration suggests localization. The implementation records redundancy and joint-over-singleton synergy signals to distinguish redundant and synergistic mechanisms.

## Cross-task overlap

Singleton causal-score vectors are compared across source tasks. A high cosine similarity suggests a shared causal substrate even when task operators differ.

## Transfer criterion

A candidate is not accepted merely because it fits source tasks. The shared structure must also satisfy the causal gates and provide a useful frozen transfer path to the related holdout task.
