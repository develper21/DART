# DART-1.3 Theory

## Hypothesis
A reusable DART primitive should be discovered from causal/relational invariants of neural computation, rather than from a learned task embedding or a task-specific residual network.

## What changes from DART-1.2
DART-1.2 used a behavioral signature followed by a learned conditioner. DART-1.3 removes learned task-conditioned machinery from the compiled primitive.

The shared core is fit jointly across meta-tasks using:
- value response of the teacher computation
- directional response under controlled perturbations
- second-order interaction response

The compiled structured core is then frozen for holdout evaluation.

## Acceptance test
A DART primitive is interesting only if it:
1. preserves useful capability on a related holdout task,
2. survives causal intervention tests,
3. uses substantially less computation than the neural control, and
4. shows greater transfer on a related task than on a contrast/unrelated task.

A positive result must not depend on task IDs, learned task codes, or large target-task residual networks.
