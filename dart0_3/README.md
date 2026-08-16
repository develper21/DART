# DART-0.3

DART-0.3 attacks the biggest remaining confound from DART-0.2:

> Does DART+adaptation beat simply training a smaller model longer?

This version uses the same task-update budget for:
- Scratch-small
- DART+Adaptation
- Distill+Adaptation

It also adds cross-task transfer:
- train replacement on a source task
- move the replacement to a different target task
- give DART and Scratch the same target adaptation budget

## Full run

```bash
python3 dart03.py
```

## Staged GPU run

```bash
python3 dart03.py \
  --seeds 1 2 \
  --tasks add compose \
  --teacher-steps 800 \
  --reference-steps 1200 \
  --replacement-steps 250 \
  --transfer-steps 400 \
  --train-size 6000 \
  --test-size 1500 \
  --trace-batches 20 \
  --latency-iters 20
```

## What counts as a strong result?

The strongest DART evidence would be:

1. DART+Adaptation >= Scratch-small at matched task-training compute.
2. DART+Adaptation >= Distill+Adaptation at matched task-training compute.
3. The result repeats over seeds.
4. The adapted DART keeps the smaller target computation.
5. A source-trained replacement provides useful transfer to a different target task.

If DART loses to Scratch at matched compute, we should NOT force the theory.
That result means the current mechanism is not special enough and should be redesigned.

## Important accounting note

This prototype uses an approximate FLOP estimator for model-training budget normalization
and exact equal numbers of task-update steps within each run. It is not a hardware-level
FLOP counter. CUDA latency is measured separately.
