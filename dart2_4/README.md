# DART-2.4 — Causal Slot Identification

DART-2.4 extends DART-2.3's interleaved shared/task factorization with an explicit **causal slot identification** objective.

## Core hypothesis

Task-specific computation may live only at a few internal locations of an otherwise shared computational skeleton. DART-2.4 asks whether those locations are **causally necessary** for cross-task differences rather than merely predictive or convenient.

Architecture:

```text
G1 -> A_t,1 -> G2 -> A_t,2 -> G3 -> A_t,3
```

where `G_i` are shared structured nodes and `A_t,i` are tiny task-specific operators.

## New DART-2.4 measurement

For every active adapter slot, the experiment removes that slot from both source-task rules and measures how much of the cross-task output-difference signal disappears.

A large drop means that slot carries causal task-specific information.

Reported as:

- `slot_causal_necessity`
- per-slot `slot_causal_scores`
- learned placement vs random placement

The old rule-causal score remains in the report as a diagnostic, but the **eligibility gate is driven by slot-level causal necessity**.

## Terminal progress monitor

Full experiments now print a live in-place progress bar:

```text
[DART-2.4][seed 1/2] [==============>               ] 52.54% | candidate-controls | 138/216 | slot_causal=0.168
```

The bar advances through:

1. teacher training
2. candidate search
3. shared fitting
4. candidate controls
5. frozen holdout transfer
6. 100% completion before the JSON result is written

## Main command

```bash
python3 dart024.py \
  --seeds 1 2 \
  --all-tasks add compose mul sub \
  --holdout-tasks sub \
  --contrast-tasks sort \
  --teacher-steps 800 \
  --core-fit-steps 300 \
  --adapter-fit-steps 120 \
  --target-adapter-fit-steps 400 \
  --transfer-control-steps 400 \
  --separate-control-steps 200 \
  --train-size 6000 \
  --verifier-size 1500 \
  --test-size 1500 \
  --rel-samples-per-task 2048 \
  --fit-batch-samples 512 \
  --max-active-adapters 2 \
  --device cuda
```

Results are saved to `dart024_results.json`.
