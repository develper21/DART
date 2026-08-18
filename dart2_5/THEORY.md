# DART-2.5 Theory

## Problem exposed by DART-2.4

DART-2.4 found positive slot-causal-necessity signals, but aggregate rule causal fidelity remained weak. Therefore, “important slot” is not enough.

## DART-2.5 hypothesis

For each active task-specific slot `s`, compare two interventions:

1. Teacher: remove the teacher's corresponding feed-forward computation at the replacement layer.
2. DART: remove only the task adapter at slot `s`.

If the resulting logit delta vectors are aligned, the slot is not merely useful: it reconstructs a teacher-grounded causal effect.

## Adaptive budget

Let `K` be the number of active task adapters. DART-2.5 searches `K ∈ {1, 2}` (or a user-limited subset) and penalizes active slots. This prevents the model from always consuming the full allowed task-specific budget.

## Eligibility

A candidate must satisfy source capability, stability, random-placement advantage, shared-vs-separate parity, operator specificity, adapter effect, slot causal necessity, and slot reconstruction fidelity thresholds.

## Expected breakthrough

A strong result would show:

- small `K`,
- high slot necessity,
- high teacher-grounded reconstruction fidelity,
- frozen shared skeleton,
- useful unseen-task transfer,
- and a clear rejection of random placement / separate-task memorization controls.
