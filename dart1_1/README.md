# DART-1.1

**Dynamic Algorithm Replacement Training — Tiny Task-Conditioned Shared Primitive**

DART-1.1 targets the transfer failure from DART-1.0. Instead of a large task-specific residual network, the replacement uses one shared structured core plus a tiny task code.

### Research question
Can a computational primitive discovered across multiple meta-tasks be reused on an unseen related task when the shared primitive is frozen and only a tiny task code is allowed to adapt?

### Run

```bash
python3 dart011.py \
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
  --task-dim 4 \
  --code-weight 0.05 \
  --routing-weight 0.20 \
  --ablation-weight 0.10 \
  --device cuda
```

### Important outputs
- `zero_shot_centroid`: frozen shared primitive with mean meta-task code.
- `zero_shot_zero_code`: frozen shared primitive with no task-conditioning signal.
- `adapted`: frozen primitive + only the tiny holdout task code adapted.
- `mlp_control`: matched neural control.
- `tiny_code_adaptation_gain_points`: improvement versus held-out teacher baseline.
- `vs_mlp_control_points`: adapted DART minus MLP control.
