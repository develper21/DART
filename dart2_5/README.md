# DART-2.5 — Causal Slot Reconstruction + Adaptive Task Budget

DART-2.5 extends DART-2.4 by requiring a selected task-specific slot to do more than be useful or causally necessary: its intervention should reproduce the **teacher's causal intervention direction**.

Core model:

```text
Shared G1 → Task Adapter 1 → Shared G2 → Task Adapter 2 → Shared G3 → Task Adapter 3
```

New ideas:

- Teacher-grounded slot reconstruction fidelity.
- Adaptive task-specific adapter budget with an explicit penalty on active slots.
- Slot necessity + slot reconstruction are both required for eligibility.
- Random placement, operator permutation, shared-vs-separate, and MLP controls remain.
- Live terminal progress reporting is retained.

The key hypothesis is:

> A reusable algorithmic variation should be localized in a small number of causally necessary slots whose intervention direction matches the teacher.

This is a research prototype, not a claim of a solved reusable-computation problem.
