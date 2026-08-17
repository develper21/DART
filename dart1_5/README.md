# DART-1.5 — Necessary-Parameter & Identifiable Primitive Discovery

DART-1.5 is the next research step after DART-1.4. Its purpose is **not** to add another operator family. It fixes a more fundamental problem exposed by DART-1.4: a factorized primitive could win even when its task parameter vector was effectively unused (`theta = 0`) and the causal metric could remain high despite weak relational agreement.

## Research question

> Can DART discover a shared structured computational mechanism whose task parameters are **necessary, stable, causally effective, and transferable**?

The intended factorization is:

```text
same shared mechanism C
        +
small explicit task parameter theta
        ↓
      task behavior
```

A valid primitive must prove that changing `theta` actually changes the computation.

## What changed from DART-1.4

DART-1.5 adds hard scientific gates instead of relying on a soft weighted score:

1. **Capability gate** — source-task performance must be meaningful.
2. **Theta norm gate** — the fitted task parameters must be non-trivial.
3. **Theta-effect gate** — perturbing `theta` must measurably change the primitive output.
4. **Relational gate** — the replacement must match teacher directional behavior.
5. **Theta stability gate** — theta fitted on disjoint source splits must remain reasonably stable.

Candidates that do not pass these checks cannot qualify as a valid DART primitive.

## DART-1.4 dead-zone fix

In DART-1.4 several structured families were zero-initialized in a way that could make both the primitive output and the gradient with respect to `theta` effectively zero at initialization. DART-1.5 initializes basis outputs non-zero and starts `theta` away from zero, so both the shared basis and task parameters receive meaningful gradients from the beginning.

## Main experimental flow

```text
Meta tasks: add + compose + mul
             ↓
     discover factorized C(x, theta)
             ↓
     fit source thetas independently
             ↓
     test theta necessity
             ↓
     test theta stability
             ↓
     test causal theta effect
             ↓
          freeze C
             ↓
      related holdout: sub
             ↓
     zero-shot + theta-only adaptation
             ↓
       contrast holdout: sort
```

`MLP` is retained only as a neural control, not as an eligible DART winner.

## Important outputs

The JSON reports:

- source-task accuracy and worst-task accuracy
- relational agreement
- theta norm
- theta effect size
- theta stability
- candidate eligibility
- zero-shot holdout performance
- theta-adapted holdout performance
- matched MLP control
- parameter and MAC counts

## Full experiment

```bash
python3 dart015.py \
  --seeds 1 2 \
  --all-tasks add compose mul sub \
  --holdout-tasks sub \
  --contrast-tasks sort \
  --teacher-steps 800 \
  --core-fit-steps 300 \
  --theta-fit-steps 120 \
  --meta-theta-adapt-steps 200 \
  --target-theta-fit-steps 400 \
  --surgery-rounds 2 \
  --transfer-control-steps 400 \
  --train-size 6000 \
  --verifier-size 1500 \
  --test-size 1500 \
  --rel-samples-per-task 2048 \
  --rel-directions 4 \
  --theta-dim 4 \
  --theta-delta 0.25 \
  --device cuda
```

## Interpretation rule

A DART-1.5 result should **not** be called a reusable primitive merely because it is small or because a composite score is high.

The stronger claim requires:

```text
non-trivial theta
+ stable theta
+ causal theta effect
+ meaningful source capability
+ frozen-core related-task transfer
```

This separation is deliberate: DART-1.5 is designed to distinguish a genuinely parameterized mechanism from a cheap but degenerate approximation.

## Current status

DART-1.5 is an experimental research version. A complete CUDA run is required before drawing conclusions about the hypothesis.
