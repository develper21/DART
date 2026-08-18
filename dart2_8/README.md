# DART-2.8

## Frozen Causal Primitive Reparameterization + Strict Transfer

DART-2.8 is the next experiment after DART-2.7. It takes the localized causal primitive identified during structural discovery and tests whether that primitive can be **frozen** and reused across tasks by fitting only a tiny task-specific reparameterization.

### Research question

Can a causally identified primitive `P*` remain frozen while a tiny structured `phi_task` reconfigures it for an unseen related task?

### Main protocol

1. Discover a shared structured candidate across source tasks.
2. Select a candidate using source capability, causal localization, cross-task overlap, and control gates.
3. Freeze the shared skeleton and primitive base state.
4. Evaluate zero-shot transfer.
5. Fit only two scalar parameters per active primitive slot (`phi_scale`, `phi_shift`).
6. Compare with full adapter fitting on the same frozen skeleton.
7. Compare against primitive permutation and random-primitive controls.
8. Compare against a matched MLP control.
9. Repeat the same frozen transfer on the unrelated contrast task.

### Important restriction

Target transfer does **not** update the discovered skeleton or primitive base state. This keeps the M4 transfer experiment strict.

### Progress output

The terminal prints a live progress bar for teacher training, primitive search, controls, and frozen transfer. On completion the final JSON is written to `dart028_results.json`.
