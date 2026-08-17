# DART-1.6

## Compositional Factorized Primitive Discovery

DART-1.6 tests whether the computational basis that DART-1.5 represented as one structured primitive can become more expressive through a **small composition of shared structured primitives**, while keeping the task-specific parameter vector explicit and low-dimensional.

### Core hypothesis

Instead of only:

`y = C(x, theta)`

DART-1.6 tests:

`y = C2(C1(x, theta1), theta2)`

The shared structure is frozen for transfer; only small explicit theta vectors are refit for a new task.

### Why this version exists

DART-1.5 established that theta can be non-zero, stable, and causally effective, but source-task capability remained limited. DART-1.6 therefore targets **primitive expressiveness**, not task conditioning.

### Research controls

- structured compositional candidates only
- no task embedding
- no large conditioner
- no task-specific residual network
- MLP used only as a control
- source capability gate
- theta-effect and theta-stability gates
- related holdout (`sub`)
- contrast holdout (`sort`)

## Full run

```bash
python3 dart016.py \
  --seeds 1 2 \
  --all-tasks add compose mul sub \
  --holdout-tasks sub \
  --contrast-tasks sort \
  --teacher-steps 800 \
  --core-fit-steps 300 \
  --theta-fit-steps 120 \
  --target-theta-fit-steps 400 \
  --transfer-control-steps 400 \
  --train-size 6000 \
  --verifier-size 1500 \
  --test-size 1500 \
  --rel-samples-per-task 2048 \
  --rel-directions 4 \
  --theta-dim 4 \
  --device cuda
```

## Interpretation

A strong DART-1.6 result requires the composed structure to outperform the best single primitive on source tasks, maintain meaningful and stable theta values, and improve frozen-structure transfer to `sub` while remaining poor on the unrelated `sort` control.
