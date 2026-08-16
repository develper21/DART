# DART-0: Dynamic Algorithm Replacement Training

## Hypothesis

A neural network can learn to identify a repeatedly-used internal computation,
synthesize a cheaper computational replacement for that sub-computation, and
continue training successfully after surgically replacing the original path.

The claim is deliberately stronger than ordinary pruning/distillation:

1. The replacement is discovered from the model's observed computation.
2. The replacement must survive counterfactual/intervention tests.
3. The original computation is actually removed from the forward path.
4. Training continues after replacement.
5. We measure capability retained per unit inference compute.

## Formal objective

Let the model be f_{\theta,G}(x), where G is its computation graph.
For an internal region S in G, construct candidate replacement R.

We want:

    R(h) ~= S(h)

for observed hidden states h and perturbations/interventions of h, while:

    Cost(R) << Cost(S)

and downstream task behavior remains within tolerance.

Candidate score:

    Score(R) =
        Retention(R)
        - lambda * RelativeCompute(R)
        - beta * FailureRate(R)
        + gamma * CounterfactualGeneralization(R)

## Acceptance gates

A candidate is accepted only if all gates pass:

A. Behavioral equivalence:
   internal-output discrepancy below tolerance.

B. Downstream capability retention:
   task loss/accuracy stays within a tolerance on held-out data.

C. Counterfactual robustness:
   replacement works on perturbed/intervened hidden states.

D. Efficiency:
   measured or estimated compute is materially lower.

E. Persistence:
   after a short adaptation phase, the replaced model remains viable.

## DART-0 scope

We intentionally keep the first experiment small:

- tiny Transformer encoder
- synthetic algorithmic tasks
- one candidate replacement site
- candidate = smaller MLP bottleneck
- teacher-subgraph distillation
- hidden-state perturbation tests
- surgical replacement + rollback

We are NOT claiming this proves the general theory. DART-0 is a falsification experiment.

## Success signal

The strongest early signal is:

    capability_after / inference_cost_after
        >
    capability_before / inference_cost_before

while the gain cannot be explained by simply training a smaller model
from scratch or by ordinary knowledge distillation alone.
