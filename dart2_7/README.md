# DART-2.7

## Causal Primitive Localization & Distributed Mediation Discovery

DART-2.7 follows the DART-2.6 finding that intervention-level necessity can be high while any single slot can appear weak. This version tests whether the causal computation is **localized, redundant, or synergistic** across a small set of internal task-specific slots.

### Research object

For every candidate shared rule graph, DART-2.7 evaluates every non-empty subset of active task slots (within the adapter budget). It records singleton, pair, and joint intervention effects, then searches for a **Minimal Causal Set (MCS)** that is necessary, sufficient, teacher-aligned, and compact.

### Main diagnostics

- Causal concentration: whether one component explains most of the joint causal effect.
- Redundancy index: whether singleton components already explain the joint effect.
- Synergy index: extra effect obtained only from a combination.
- Minimal causal set size: smallest subset meeting the intervention-fidelity threshold.
- Cross-task causal overlap: whether source tasks share the same causal structure.
- Random causal-set gap: learned causal sets vs same-size random sets.
- Frozen `sub` transfer and `sort` contrast.

### Terminal presentation

The CLI prints a large `DART-2.7` banner with the supplied DART emblem rendered to the right on the same rows, followed by the live progress bar used in earlier versions. The original `dart_logo.png` is included in the package.

### Research command

```bash
python3 dart027.py \
  --seeds 1 2 \
  --all-tasks add compose mul sub \
  --holdout-tasks sub \
  --contrast-tasks sort \
  --teacher-steps 800 \
  --core-fit-steps 300 \
  --adapter-fit-steps 120 \
  --target-adapter-fit-steps 400 \
  --transfer-control-steps 400 \
  --separate-control-steps 200 \
  --train-size 6000 \
  --verifier-size 1500 \
  --test-size 1500 \
  --rel-samples-per-task 2048 \
  --fit-batch-samples 512 \
  --causal-probe-size 64 \
  --max-active-adapters 2 \
  --device cuda
```
