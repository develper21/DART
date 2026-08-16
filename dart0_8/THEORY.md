# DART-0.8 — Routing-Preserving Replacement

## Motivation

DART-0.6 replaced an entire 3-block trajectory with one repeatedly applied
operator. DART-0.7 added small residual adapters, but both still removed the
original Transformer attention pathway. Both runs produced strong trajectory
consistency but severe capability loss.

DART-0.8 therefore isolates one hypothesis:

> **The computation can only be safely replaced if the model's information-routing
> pathway is preserved.**

## Hypothesis

Keep each block's original:

- LayerNorm 1
- multi-head attention parameters
- residual connection around attention
- LayerNorm 2

Replace only the feed-forward transformation with:

`FF_i(z)  ->  C(z) + R_i(z)`

where `C` is shared across the trajectory and `R_i` is a tiny block-specific
residual adapter.

This is deliberately less aggressive than DART-0.6/0.7. The experiment asks
whether the catastrophic capability loss was caused primarily by destroying
information routing rather than by the idea of shared computation itself.

## Causal routing check

The experiment records teacher and candidate attention maps and computes a
routing agreement score. It also performs an attention-ablation test to measure
how much task accuracy changes when the retained routers are removed.

This follows the spirit of causal activation patching / component intervention:
the point is not to infer that a state is important merely because it resembles
the teacher, but to test whether changing a causal pathway changes behaviour.

## Interpretation

### Strong positive signal

A compelling DART-0.8 outcome would be:

1. capability remains close to teacher,
2. FF compute is materially lower,
3. routing agreement remains high,
4. adaptation does not need a large recovery,
5. related-task transfer is at least competitive with scratch.

### Negative signal

If routing is preserved but capability still collapses, then the hypothesis
"routing was the main missing ingredient" is weakened. The next research step
should then examine token-to-token and feature-level causal subgraphs rather
than whole-FF replacement.

## Important limitation

This experiment is still a tiny synthetic Transformer. A positive result would
be a mechanism-level signal, not evidence of large-model generality.
