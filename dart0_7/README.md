# DART-0.7

Shared Computational Core + Minimal Residual.

## Run

```bash
python3 dart07.py \
  --seeds 1 2 \
  --tasks add compose \
  --teacher-steps 800 \
  --core-fit-steps 300 \
  --adaptation-steps-per-round 400 \
  --surgery-rounds 2 \
  --transfer-adaptation-steps 400 \
  --train-size 6000 \
  --verifier-size 1500 \
  --test-size 1500 \
  --trajectory-batches 20 \
  --verifier-batches 20 \
  --residual-rank 2 \
  --residual-weight 0.05 \
  --device cuda
```

The output is `dart07_results.json`.

## Main comparison

DART-0.7 compares shared computational cores plus minimal residual adapters. The main control is a shared small MLP core with the same residual budget.

The experiment records:

- teacher accuracy/loss
- pre/post adaptation accuracy/loss
- core parameters/MACs
- residual parameters/MACs
- residual fraction
- trajectory consistency
- full-model CUDA latency
- transfer to related tasks

## Research discipline

A low-dimensional core is not automatically an algorithm. The strongest evidence remains capability preservation plus repeated cross-task transfer under the same evaluation budget.
