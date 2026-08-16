# DART-0.9

Routing-Preserving Structured Replacement with Neural Controls.

## Run

```bash
python3 dart09.py \
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
  --residual-weight 0.01 \
  --routing-weight 0.20 \
  --ablation-weight 0.10 \
  --device cuda
```

Output is `dart09_results.json` by default.

## Important interpretation

The MLP is a **control**, not a DART candidate. The DART winner is always selected only from structured operators.

The JSON contains `kind` and `eligible` fields for candidate records so the distinction is explicit.
