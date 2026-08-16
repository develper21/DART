# DART-0.2

## Purpose

DART-0.2 is a validation release. It does not add a new architectural trick.
It tests whether DART-0.1's signal survives:

- 5 random seeds
- multiple synthetic tasks
- adaptation budgets
- accuracy/loss tracking
- target-subgraph compute tracking
- CUDA latency (when CUDA is available)

## Full run

```bash
python3 dart02.py
```

This is intentionally more expensive than DART-0.1 because it runs:

- 5 seeds
- 5 tasks
- 4 baseline-ish training paths
- 6 adaptation points

For a first smoke run:

```bash
python3 dart02.py \
  --seeds 1 2 \
  --tasks add compose \
  --baseline-steps 400 \
  --scratch-steps 400 \
  --replacement-steps 200 \
  --adaptation-steps 0 100 200 400 \
  --train-size 4000 \
  --test-size 1000 \
  --trace-batches 20 \
  --latency-iters 20
```

## Output

Results are saved to `dart02_results.json`.

## Interpretation

The main signal is NOT merely whether DART reaches high accuracy.

We want to know whether, across seeds and tasks:

1. DART+Adaptation recovers the teacher capability reliably.
2. Its variance is acceptable.
3. It beats or matches Scratch-small and Distill-small at lower target FF compute.
4. The recovery curve has a systematic shape as adaptation budget increases.

If DART only wins on one seed/task, the hypothesis is not yet robust.
