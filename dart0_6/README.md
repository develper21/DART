# DART-0.6

## Trajectory Compression / Reusable Transition Discovery

DART-0.6 is the direct continuation of DART-0.5 and follows the research
backup's final direction.

### What changed

DART-0.5 repeatedly replaced one FFN. DART-0.5's `small_mlp` won all eight
candidate selections, showing that changing the replacement objective was not
enough to discover a non-neural computational primitive.

DART-0.6 freezes **single-layer replacement as the main unit** and instead
studies a three-block trajectory:

```text
h0 -> h1 -> h2 -> h3
```

It searches for one shared transition `O` that can explain all three steps and
then applies `O` three times as the surgical replacement.

### Run

```bash
python3 dart06.py \
  --seeds 1 2 \
  --tasks add compose \
  --teacher-steps 800 \
  --operator-fit-steps 300 \
  --adaptation-steps-per-round 400 \
  --surgery-rounds 2 \
  --transfer-adaptation-steps 400 \
  --train-size 6000 \
  --verifier-size 1500 \
  --test-size 1500 \
  --trajectory-batches 20 \
  --verifier-batches 20 \
  --latency-iters 30
```

### Primary measurements

- teacher vs DART downstream accuracy/loss
- trajectory consistency of the shared transition
- replacement parameters and MACs
- post-surgery adaptation recovery
- CUDA latency
- repeated surgery winners
- source → target transfer

### Research interpretation

If `shared_mlp` dominates again, the evidence will suggest that even a
multi-layer trajectory can be compressed most effectively by a learned neural
operator.

If a compact non-neural transition wins and transfers across related tasks,
that is a stronger signal for reusable computational structure.

This repository intentionally does not claim that either result proves a new
learning paradigm.
