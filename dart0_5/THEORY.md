# DART-0.5 Theory

## Hypothesis

A model may contain an internal computation whose exact hidden representation
is not itself important. What matters is the downstream behavior that the
computation enables.

Therefore a replacement should be selected by:

    downstream_task_loss
    + intervention_robustness
    + complexity_cost

rather than by:

    hidden_output_MSE

## Independent verifier

The candidate generator is not its own judge.

Verifier data are held out from candidate fitting. The verifier measures:
- downstream accuracy/loss;
- final prediction consistency when the target computation is internally
  perturbed.

This reduces the chance that a candidate wins only because it memorized its
training traces.

## Falsifiers

The DART-0.5 hypothesis weakens if:
- small MLP always wins;
- non-neural operators do not transfer;
- DART offers no gain over scratch/distillation controls;
- adaptation does not reliably recover post-surgery capability.

No novelty claim is made until these controls are passed.
