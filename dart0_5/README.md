# DART-0.5

## Main idea

DART-0.4 still selected a small MLP because the replacement was effectively
learning to imitate the teacher's hidden computation.

DART-0.5 changes the objective:

    DO NOT MATCH HIDDEN OUTPUTS.
    MATCH DOWNSTREAM BEHAVIOR.

A candidate operator is inserted into a frozen teacher body and optimized
directly with task loss. Then an independent verifier evaluates:

- held-out downstream accuracy/loss
- stability of the final prediction under internal perturbations

Candidates:
- identity
- diagonal affine
- polynomial
- signed affine
- low-rank linear
- small MLP (control only)

The system then performs graph surgery and full-model adaptation.

## Recommended staged run on the GPU

```bash
python3 dart05.py \
  --seeds 1 2 \
  --tasks add compose \
  --teacher-steps 800 \
  --operator-fit-steps 250 \
  --adaptation-steps-per-round 400 \
  --surgery-rounds 2 \
  --transfer-adaptation-steps 400 \
  --train-size 6000 \
  --verifier-size 1500 \
  --test-size 1500 \
  --verifier-batches 20 \
  --latency-iters 30
```

## What to look for

Most important:
1. Does a non-MLP operator ever win?
2. Does a winner survive independent behavioral verification?
3. Does post-surgery adaptation recover or improve capability?
4. Does round 2 discover another replacement?
5. Does a source-task operator provide transfer to a related target?
6. Are CUDA latency numbers positive and reproducible?

If `small_mlp` wins every round again, that is a useful negative result:
behavioral search still prefers neural approximators, and DART has not yet
demonstrated a distinct computational invention mechanism.

If compact non-MLP operators win and transfer, that is much stronger evidence
that the system is learning reusable computational primitives.
