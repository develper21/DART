# DART-2.0 Theory

## 1. Motivation

DART began with a simple idea: a trained neural model may repeatedly perform computation that could potentially be replaced by a cheaper reusable procedure.

The early experiments established that replacement and adaptation are possible. Later versions showed that preserving routing and moving beyond small MLP substitutes matters. The main unresolved problem became abstraction: how to discover something that is truly reusable rather than a task-specific approximation.

DART-1.8 showed that a compact bottleneck can preserve information without proving causal necessity. DART-1.9 then tested direct latent/state interchangeability and obtained a negative swap-margin signal in its full run. DART-2.0 treats those failures as evidence that the reusable object should be the **transformation rule**, not the state itself.

## 2. Hypothesis

For related tasks t, there exists a shared structured rule graph G and a small task parameter vector θ_t such that:

```text
y_t ≈ G(x, θ_t)
```

The graph should be reusable while θ changes with the task.

A successful result requires more than source-task fit. The graph must also:

1. have meaningful task-dependent parameters,
2. react causally when rule nodes are perturbed,
3. outperform a random graph with the same budget,
4. distinguish correct theta assignment from permuted assignment,
5. transfer to an unseen related task after the graph is frozen.

## 3. Rule motifs

DART-2.0 currently tests shallow graph motifs:

### Sequential

```text
x → A(x) → B(A(x))
```

### Parallel sum

```text
x → A(x) ─┐
          ├→ x + A(x) + B(x)
 x→ B(x) ─┘
```

### Residual parallel

```text
x + A(x) + B(x) + optional interaction term
```

These motifs are deliberately small. The experiment should establish whether a reusable rule graph exists before increasing search depth.

## 4. Task parameterization

Each graph node is gated by a small explicit theta value. For a two-node graph:

```text
G(x,θ1,θ2)
```

The same graph is fit jointly across source tasks while theta values are learned separately.

The theta vector is not intended to become a hidden task embedding or a large residual network. It is intended to be a compact configuration of a shared rule.

## 5. Rule-level causal intervention

A rule graph is meaningful only if its internal operations matter.

For a node i, DART-2.0 can disable or perturb that node and measure the resulting downstream behavior.

The observed change is compared with the rule graph's predicted local change. This is a stronger requirement than merely checking whether theta moves the primitive output.

## 6. Theta permutation control

Let the source-task parameters be θ_A, θ_B, θ_C. A correct assignment should outperform a permutation such as:

```text
A → θ_B
B → θ_C
C → θ_A
```

If there is little difference, theta is probably not carrying meaningful task configuration.

## 7. Random graph control

A graph with the same topology and parameter budget is freshly initialized and tested with the learned theta values.

The learned graph should outperform this random control if the graph itself contains a discovered computational structure.

## 8. Transfer

After source discovery:

```text
freeze G
```

For the holdout related task:

```text
fit only θ_holdout
```

No graph retraining, conditioner, task embedding, or large residual is allowed.

This is the cleanest test of reusable computation in this experimental family.

## 9. Interpretation of failure

DART-2.0 separates failure into:

- `RULE_FAIL`: the graph cannot explain the source tasks.
- `PARAM_FAIL`: theta does not meaningfully configure the shared graph.
- `CAUSAL_FAIL`: rule interventions do not match teacher behavior.
- `PERMUTATION_FAIL`: task-to-theta binding is not meaningful.
- `RANDOM_GRAPH_FAIL`: the learned graph is not better than a same-budget random graph.
- `TRANSFER_FAIL`: the graph works on source tasks but not on an unseen related task.
- `SPECIFICITY_FAIL`: the graph works similarly on the unrelated contrast task.

## 10. What would constitute strong evidence?

The strongest DART-2.0 result would look like:

```text
same G
+
small task-specific θ
+
strong source performance
+
positive rule-intervention fidelity
+
correct theta pairing >> shuffled pairing
+
learned graph >> random graph
+
G frozen
+
θ_sub recovers unseen sub
+
sort remains poor
```

That would move the work substantially closer to the original DART objective: discovering and reusing a computational rule rather than merely approximating a neural function.
