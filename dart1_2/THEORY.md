# DART-1.2 Theory — Behavioral-Invariant Primitive Discovery

## Motivation
DART-1.0 jointly discovered a structured core across meta-tasks, but the holdout task did not transfer. DART-1.1 removed large task-specific residuals and replaced them with a 4D trainable task code; transfer still failed.

## DART-1.2 hypothesis
The failure may be upstream of task conditioning: the system is discovering a primitive from task identity rather than from task-invariant behavior.

DART-1.2 therefore removes task IDs and trainable task codes. Each task is represented by a deterministic **behavioral signature** computed from fixed probe input/output pairs. The signature contains output-distribution statistics, controlled first-order response statistics, and a two-variable interaction statistic.

The same structured computational core is jointly fit across meta-tasks. The shared core and behavior conditioner are then frozen for the holdout task. The holdout signature is computed from its observed behavior without gradient-based code learning. A second condition-only adaptation experiment is reported separately.

## Falsification
A positive result requires all of the following:
1. useful holdout zero-shot capability from the frozen shared primitive;
2. competitive capability after conditioner-only adaptation;
3. no need for a large task-specific residual network;
4. evidence that the same primitive is selected across meta-tasks;
5. competitive performance against a matched MLP control.

A negative result means the task-invariant abstraction is not captured by the current behavioral signature/core interface.
