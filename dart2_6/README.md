# DART-2.6 — Minimal Causal Mediation Discovery

DART-2.6 extends DART-2.5 from slot reconstruction to **minimal causal mediation**.

## Research hypothesis
A reusable computational primitive should be:

1. **Necessary** — ablating the candidate changes the learned computation.
2. **Sufficient** — restoring/intervening through the candidate reproduces the teacher-aligned effect.
3. **Trajectory-faithful** — intervention strength follows a teacher-aligned causal response curve.
4. **Minimal** — the smallest intervention that reaches the fidelity threshold should be preferred.
5. **Transferable** — the validated causal structure should remain useful for an unseen related task.

## New DART-2.6 metrics
- `mediation_necessity`
- `mediation_sufficiency`
- `trajectory_fidelity`
- `minimal_alpha`
- `minimality`
- `cme` (Causal Mediation Efficiency)

The acceptance gate requires all core mediation criteria rather than allowing raw source accuracy or reconstruction fidelity alone to declare success.

## Progress reporting
The CLI prints a live terminal progress bar with seed, phase, candidate count, and current mediation metrics. At 100%, final summaries are printed and the JSON result is written.
