# DART-0.8

## Routing-Preserving Shared Computation

DART-0.8 targets the failure observed in DART-0.6 and DART-0.7: replacing the
whole trajectory removes the original model's information-routing structure.

The new replacement keeps every original attention router and replaces only the
FFN part of the selected block span with a shared computational core plus small
step-specific residual adapters.

## Run

```bash
python3 dart08.py \
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

## Key fields

- `routing_agreement`: similarity of candidate vs teacher attention maps.
- `attention_ablation_drop`: accuracy lost when the retained attention routers
  are causally removed.
- `replace_params` / `replace_macs`: cost of the new FF path.
- `transfer_gain_points`: DART accuracy minus matched scratch accuracy on the
  target task.

Do not interpret routing agreement by itself as proof of causal equivalence.
The ablation test is included specifically to add a causal check.
