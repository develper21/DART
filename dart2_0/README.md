# DART-2.0

## Computational Rule-Graph Discovery + Frozen Rule Transfer

DART-2.0 is the next stage of the DART research program.

The central shift is from **state/representation transfer** to **computational rule transfer**.

Earlier DART versions progressively tested:

- cheaper neural replacement,
- capability preservation,
- structured primitive discovery,
- explicit task parameters,
- compositional primitives,
- typed intermediate representations,
- causal bottlenecks, and
- counterfactual state interchangeability.

DART-1.9 showed that a representation can fit source computation well while a proposed cross-task state swap is actually **less faithful than a random swap**. That means a reusable computation should not be defined only as a reusable latent state.

DART-2.0 therefore asks:

> **Can DART discover a shared structured transformation rule graph, configure that graph with a tiny explicit task parameter vector, freeze the graph, and reuse it on an unseen related task without retraining the graph?**

---

## Core idea

Instead of transferring a state:

```text
Task A state  →  Task B state
```

DART-2.0 attempts to transfer a rule:

```text
R_add
R_mul
R_compose
   ↓
shared rule graph G
   ↓
θ_task
   ↓
unseen task
```

The rule graph is composed from small structured operators such as:

- diagonal
- polynomial
- affine-polynomial
- low-rank

Candidate graph motifs currently include:

```text
sequential
parallel_sum
residual_parallel
```

The graph `G` is shared across source tasks. Only a small explicit `theta` vector is task-specific.

---

## Architecture

The replacement inside the preserved routing path is conceptually:

```text
hidden state x
     ↓
shared rule graph G(x, θ)
     ↓
replacement output
```

A graph can contain multiple structured nodes:

```text
x
│
├── Rule A ──┐
│            ├── combine → output
└── Rule B ──┘
```

The current implementation supports shallow rule motifs rather than an unrestricted program search. This keeps the experiment interpretable and prevents a huge combinatorial search space from becoming the experiment itself.

---

## Research controls

DART-2.0 does not accept a candidate merely because it is small.

Every candidate is tested against multiple controls:

### 1. Source capability gate

The shared rule graph must achieve a minimum average and worst-task source accuracy.

### 2. Theta stability

The fitted task parameters must be reasonably stable when fitted on separate halves of the source data.

### 3. Rule-level causal intervention

A graph node is disabled or perturbed and the resulting downstream teacher behavior is compared with the rule graph's predicted intervention effect.

### 4. Theta-permutation control

Correct task-to-theta assignments are compared against a deliberately permuted assignment.

A meaningful rule should satisfy:

```text
correct pairing > permuted pairing
```

### 5. Random-graph control

A fresh graph with the same topology and parameter budget is evaluated.

A learned shared rule should outperform the random graph:

```text
learned graph > random graph
```

### 6. Unseen-task transfer

The rule graph is frozen. Only a tiny target-task theta vector may adapt.

### 7. Contrast task

An unrelated task such as `sort` is used to test whether the rule graph is genuinely task-family-specific.

---

## Experimental protocol

Default source tasks:

```text
add
compose
mul
```

Related holdout:

```text
sub
```

Contrast:

```text
sort
```

The protocol evaluates:

```text
Teacher
DART zero-shot
DART theta-adapted
Theta permutation control
MLP control
```

The main research question is not whether DART beats every neural control immediately. It is whether a **single discovered rule graph** remains useful when moved to an unseen related task with only a small explicit configuration change.

---

## Success ladder

### Level 1 — Shared rule

The same rule graph works across multiple source tasks.

### Level 2 — Task configuration

Different source behaviors are explained by different theta values using the same graph.

### Level 3 — Rule intervention fidelity

Changing/disabling a graph component produces a teacher-consistent behavioral change.

### Level 4 — Frozen transfer

The graph is frozen and only target theta is fitted on the unseen related task.

### Level 5 — Reusable computational primitive

The frozen rule graph approaches teacher capability while remaining substantially cheaper than the neural replacement control and does not transfer equally well to the unrelated contrast task.

---

## Current DART research status

The broader DART program has established strong evidence for:

- meaningful compute reduction,
- capability recovery after replacement,
- structured/non-MLP replacement candidates,
- active explicit task parameters, and
- increasingly structured intermediate representations.

The unresolved problem is the central one:

> **discovering a genuinely reusable computational abstraction.**

DART-2.0 addresses that problem by making the **rule itself** the primary research object.

---

## Reproducibility

Example full experiment:

```bash
python3 dart020.py \
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
  --fit-batch-samples 512 \
  --device cuda
```

A research result should only be interpreted from a full CUDA run. The included smoke test is an implementation check, not a scientific result.

---

## Repository layout

```text
dart2_0/
├── dart020.py
├── README.md
├── THEORY.md
└── CHANGELOG.md
```

---

## Research principle

DART is not intended to be merely a smaller neural network.

The long-term objective is:

```text
observe computation
      ↓
discover reusable rule
      ↓
separate shared mechanism from task configuration
      ↓
freeze the mechanism
      ↓
reuse it on new related tasks
```

DART-2.0 is the first version in the series whose primary object of discovery is explicitly the **computational rule graph** rather than a latent state or representation.
