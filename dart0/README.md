# DART-0 Prototype

## Run

```bash
python dart0.py
```

For a faster smoke test:

```bash
python dart0.py --train-steps 100 --replacement-steps 100
```

## What this prototype tests

It trains a tiny Transformer, selects one FFN block as the candidate sub-computation,
learns a cheaper bottleneck replacement, tests it on perturbed hidden states, surgically
replaces the original FFN, and evaluates task retention.

## Important limitation

This first prototype is intentionally conservative. It does not yet discover arbitrary
subgraphs or invent arbitrary operator types. The first scientific question is whether
the replacement loop produces a reproducible compute/quality trade-off at all.

The next experimental milestones are:

1. candidate families beyond a small MLP;
2. automatic subgraph discovery;
3. adversarial counterexample generation;
4. continued training after surgery;
5. compare against pruning, distillation, and a small-from-scratch baseline.
