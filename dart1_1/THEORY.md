# DART-1.1 Theory

## Why 1.1 exists
DART-1.0 jointly discovered a structured primitive across `add`, `compose`, and `mul`, but its replacement-side budget still contained 1,440 task-specific residual parameters. The shared core therefore did not have to carry the full task abstraction, and frozen transfer to `sub` failed.

## DART-1.1 hypothesis
A reusable primitive should be expressed as:

`shared computational core C + tiny task code z`

rather than:

`shared core C + large task-specific residual network R_task`.

The task code is a low-dimensional vector (default 4 scalars). A shared conditioner maps `z` to a small affine modulation of the structured core output. During unseen-task transfer, the shared core **and its conditioner are frozen**. Only `z_holdout` is trainable.

## Critical falsification test
DART-1.1 is interesting only when all of the following move together:

1. held-out task capability approaches the teacher;
2. tiny-code adaptation produces meaningful recovery;
3. task-code size stays tiny relative to the replaced computation;
4. the shared core itself remains frozen during transfer;
5. performance is competitive with or better than a matched neural control.

If the shared core has low zero-shot performance and the tiny code cannot recover it, the abstraction is not transferable under this model.

## Controls
- Structured DART candidates: identity, diagonal, polynomial, affine-polynomial, low-rank.
- Neural MLP control is reported but never eligible as the DART winner.
- Zero-shot is evaluated with both the mean meta-task code (centroid) and a zero code.
- Unseen-task adaptation is restricted to the tiny task code only.
