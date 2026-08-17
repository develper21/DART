# DART-1.2

DART-1.2 tests whether the reusable primitive should be discovered from **behavioral invariants** rather than task identity or a learned task code.

## Key change from DART-1.1
- No trainable task code.
- No task-specific residual network.
- Deterministic behavioral signature from probe input/output behavior.
- Shared structured core + shared conditioner jointly fit on meta-tasks.
- Holdout uses the same frozen core and its deterministic behavior signature.

## Run
```bash
python3 dart012.py \
  --seeds 1 2 \
  --all-tasks add compose mul sub \
  --holdout-tasks sub \
  --teacher-steps 800 \
  --core-fit-steps 300 \
  --adaptation-steps-per-round 400 \
  --surgery-rounds 2 \
  --transfer-adaptation-steps 400 \
  --train-size 6000 \
  --verifier-size 1500 \
  --test-size 1500 \
  --signature-probes 64 \
  --routing-weight 0.20 \
  --ablation-weight 0.10 \
  --device cuda
```

## Primary metrics
- `zero_shot_behavior`: frozen-core holdout capability using only the behavioral signature
- `after_signature_conditioner_adaptation`: condition-only adaptation
- `mlp_control`: matched neural control
- `zero_shot_gain_points`, `adapted_gain_points`, `vs_mlp_control_points`

The smoke test is not a research result. It only verifies the pipeline.
