# DART — Dynamic Algorithm Replacement Training

> **Research status:** Experimental / in active development  
> **Current implementation frontier:** DART-1.2 (implemented; full research validation pending)  
> **Last fully validated research version:** DART-1.1

DART is a research framework for testing a simple but ambitious idea:

> **Can a neural network discover that part of its learned neural computation is no longer necessary, replace that computation with a cheaper structured procedure, preserve capability after surgery and adaptation, and eventually reuse the discovered computation across related tasks?**

The long-term objective is **not merely model compression**. Compression is one measurable benefit. The deeper goal is to investigate whether learned neural computation can be transformed into **reusable computational primitives**.

---

## 1. Why DART exists

The research started from a broader idea: repeated experience might eventually be converted into compact reusable procedures rather than being reconstructed repeatedly through neural computation.

A prior-art check showed that reusable skills, program synthesis, memory systems, and skill libraries already form a crowded area. The research therefore narrowed to a more specific question:

> **How can an AI discover that a piece of its own neural computation is no longer necessary?**

That question became DART: **Dynamic Algorithm Replacement Training**.

The original DART proposal was:

```text
train / observe neural computation
            ↓
identify repeatedly used computation
            ↓
discover a cheaper replacement
            ↓
verify behavior and robustness
            ↓
surgically replace the original path
            ↓
adapt the surrounding model
            ↓
measure capability + compute
            ↓
repeat
```

The important distinction is that DART is **not intended to be a skill library**. It is a **computation-replacement mechanism**.

---

## 2. The core hypothesis

Let:

- `f_θ(x)` = an expensive learned neural computation
- `g_φ(x)` = a cheaper candidate computational replacement

DART asks whether we can find:

```text
f_θ(x)  ≈  g_φ(x)
```

while also satisfying stronger conditions:

1. **Behavioral equivalence** — the replacement performs the relevant computation.
2. **Downstream capability retention** — the whole model remains useful after surgery.
3. **Counterfactual / intervention robustness** — the replacement does not only match ordinary examples.
4. **Lower computation** — the replacement is materially cheaper.
5. **Persistence after adaptation** — capability remains after the surrounding network adapts.
6. **Cross-task reuse** — the discovered computation should eventually work on related tasks without simply relearning the computation.

The first five are experimentally measurable today. The sixth is the main unresolved research target.

---

## 3. What DART is trying to solve

### Problem A — Neural computation is expensive

A trained network may repeatedly execute the same learned transformation. If a cheaper equivalent computation can be discovered, that work could potentially be compiled into a smaller computational primitive.

### Problem B — Compression alone is not enough

A smaller neural network can be trained or distilled directly. If DART merely produces another small neural approximator, it has not demonstrated the deeper research idea.

### Problem C — Replacement can destroy capability

Replacing a subgraph can remove information routing or intermediate representations that other parts of the network rely on. DART therefore treats surgical replacement, adaptation, routing preservation, and intervention testing as separate research concerns.

### Problem D — A useful replacement should be reusable

The long-term target is not a task-specific compression artifact. The stronger objective is:

```text
Task A + Task B + experience
        ↓
 discover reusable computation C
        ↓
 freeze C
        ↓
 Task D
        ↓
 reuse C with little / no relearning
```

This cross-task transfer property is **not yet achieved**.

---

## 4. Why DART could be useful

If the hypothesis eventually works, DART could provide several benefits:

### Compute reduction

Repeated neural computation could be replaced with smaller structured operators.

### Parameter reduction

The replacement can require substantially fewer parameters than the original neural subgraph.

### Potentially better interpretability

A structured operator such as an affine, low-rank, polynomial, piecewise, or symbolic composition is easier to inspect than an opaque high-dimensional neural block.

### Adaptive model evolution

DART is designed as an iterative process:

```text
G₀ → G₁ → G₂ → G₃ → ...
```

where the computational graph can change as useful replacements are discovered and validated.

### Reusable computation

The most important potential benefit is not compression itself. It is the possibility that a discovered computation could become a reusable primitive for future tasks.

**Important:** these are research motivations and potential benefits, not claims that DART has already achieved all of them.

---

## 5. Research architecture evolution

DART has deliberately changed its replacement unit as each experiment exposed a failure mode.

## DART-0 — Proof of concept

Initial experiment:

```text
Tiny Transformer
   ↓
trace target FFN
   ↓
fit smaller replacement
   ↓
counterfactual test
   ↓
surgical replacement
   ↓
evaluate
```

The first prototype reduced the target FFN from **8,352 parameters to 2,112** and reduced target FF computation from **8,192 to 2,048 MACs/token**. The post-surgery accuracy remained high on the tiny synthetic task.

**Lesson:** neural computation can be surgically replaced in principle, but this is not yet evidence of algorithm discovery.

## DART-0.1 — Controls + adaptation

Added:

- scratch-small control
- distillation control
- DART
- DART + post-surgery adaptation
- latency / FLOP measurement

The important result was that adaptation could restore capability after replacement.

**Problem exposed:** DART was still not separated from ordinary small-model training or distillation.

## DART-0.2 — Robustness across seeds/tasks

Expanded the experiment to more seeds, more tasks, and explicit adaptation curves.

The adaptation-recovery signal repeated strongly.

**Problem exposed:** DART received additional training during adaptation, creating a compute-budget confound.

## DART-0.3 — Compute-matched controls + transfer

Introduced:

- compute-matched task-update budgets
- scratch-small
- distill + adapt
- DART + adapt
- source → target transfer

`add → compose` showed a positive transfer signal, while `mul → sub` did not.

**Problem exposed:** DART and distillation remained too similar, and transfer appeared task-relationship dependent.

## DART-0.4 — Intervention-tested operator discovery

Introduced multiple candidate families and intervention tests:

- diagonal affine
- polynomial
- low-rank
- linear
- small MLP

and interventions such as:

```text
h
h + εd
h - εd
0.5h
1.5h
masked(h)
permuted(h)
```

The loop became:

```text
observe
 → operator discovery
 → surgery
 → adaptation
 → observe again
 → next surgery
```

**Problem exposed:** the small MLP won every tested round. The search was still behaving like hidden-output approximation / distillation.

## DART-0.5 — Behavioral replacement

Changed the replacement objective from hidden-tensor imitation toward downstream behavioral optimization with an independent verifier and non-neural candidates.

Despite the change, **small MLP still won 8/8 selections**.

**Major conclusion:** changing the replacement loss was not enough. The problem was probably the **unit being replaced**.

## DART-0.6 — Trajectory compression

The replacement unit changed from a single FFN to a multi-layer trajectory:

```text
x → h₁ → h₂ → h₃ → answer
```

The hypothesis was that a reusable operation might span several layers.

Result:

- trajectory consistency became high
- capability collapsed
- transfer remained negative

**Lesson:** reproducing a recurring latent trajectory is not the same as preserving the computation required by the task.

## DART-0.7 — Shared core + residual

Introduced:

```text
shared core C
+
small step-specific residual Rᵢ
```

The intention was to preserve a common computation while allowing small task-specific corrections.

Result:

- MLP again became the winner 8/8
- capability remained very low

**Lesson:** a residual path can become an escape hatch for ordinary neural approximation, and pure shared replacement is too aggressive.

## DART-0.8 — Routing-preserving replacement

This was the first major recovery.

Instead of removing the information-routing stack, DART preserved the original attention / routing pathway and replaced the expensive computation around it.

This change brought capability back close to the teacher:

```text
Teacher mean accuracy: 65.72%
DART-0.8 mean accuracy: 63.68%
```

That is roughly **96.9% of teacher capability** while using substantially fewer parameters in the tested model.

**Major lesson:** preserving information routing is critical for safe computation replacement in this experimental setup.

## DART-0.9 — Structured replacement + neural controls

DART-0.9 froze the successful routing-preserving architecture and removed MLPs from the DART winner pool. MLPs remained only as controls.

Structured candidates included:

- identity
- diagonal
- polynomial
- affine-polynomial
- low-rank

Result:

```text
affine_polynomial = 7/8 winners
low_rank          = 1/8 winners
```

This was the first major evidence that DART can select a **non-MLP structured computational replacement**.

However, cross-task transfer remained negative.

**Lesson:** local structured replacement works, but a locally useful structured primitive is not automatically task-invariant.

## DART-1.0 — Joint shared primitive discovery

The next step was to discover a shared primitive jointly across multiple meta-tasks:

```text
add + compose + mul
        ↓
shared structured primitive
        ↓
freeze
        ↓
held-out sub
```

Result:

- a common `affine_polynomial` primitive could be discovered
- meta-task performance improved after adaptation
- unseen-task transfer failed

For the held-out `sub` task, the shared-core zero-shot result was about **11.33%**, compared with a teacher around **17.97%** and an MLP control around **17.70%** in the aggregate experiment.

**Lesson:** joint discovery can find a common structured operator, but the discovered commonality was not sufficient to transfer to the unseen task.

## DART-1.1 — Tiny task-conditioned shared primitive

DART-1.0 still used a sizeable task-specific residual budget. DART-1.1 replaced that with a tiny learned task code:

```text
shared structured core
+
shared conditioner
+
4-dimensional task code
```

The task-specific code had only a few trainable parameters rather than a large residual network.

Result on held-out `sub`:

```text
Teacher              16.70%
Zero-shot             10.00%
Tiny-code adapted     10.93%
MLP control           23.32%
```

**Major conclusion:** the large task-specific residual was not the main cause of transfer failure. Even with a tiny task code, the shared abstraction did not transfer.

## DART-1.2 — Behavioral / relational invariant discovery

DART-1.2 is implemented but **not yet fully validated by a research run**.

The conceptual change is to stop representing a task with an explicit learned task ID/code. Instead, the replacement is conditioned on a deterministic **behavioral signature** extracted from task behavior.

The intended flow is:

```text
observe behavior
      ↓
extract behavioral / relational signature
      ↓
discover shared structured primitive
      ↓
freeze primitive
      ↓
derive signature for unseen task
      ↓
reuse primitive
```

The research question is whether the true reusable abstraction is visible in the **behavior of the computation**, rather than in a task-specific learned code.

---

## 6. What has been achieved so far

## Strongly demonstrated

- Surgical replacement of learned neural computation is feasible in the tiny experimental setup.
- Post-surgery adaptation can recover or improve capability.
- Large target-computation reductions are achievable.
- Intervention-aware candidate search can be implemented.
- Routing preservation can prevent severe capability collapse.
- Structured non-neural candidates can win over neural candidates when MLPs are excluded from the DART selection pool.
- The research loop can use failures to refine the replacement unit and experimental protocol.

## Demonstrated only partially

- Cross-task transfer: positive signals appeared for some task relationships in early versions, especially `add → compose`, but later controlled experiments showed that transfer is not reliable or universal.
- DART versus ordinary distillation: the experiments have narrowed the difference, but a general superiority claim is not established.
- End-to-end speedup: target computation and parameter reductions are clear, but the experiments do not establish a general full-model latency improvement.

## Not yet demonstrated

- A genuinely task-invariant reusable computational primitive.
- Reliable zero-shot transfer of a discovered primitive to an unseen related task.
- A proof that DART discovers an algorithm rather than a structured approximation.
- General superiority over strong neural compression / distillation baselines.

---

## 7. Current evidence snapshot

The most important measured facts from the research so far are:

| Finding | Evidence | Status |
|---|---|---|
| Target FF computation can be reduced from 8,192 → 2,048 MACs/token | DART-0 / 0.1 / 0.5 family | ✅ |
| Post-surgery adaptation can restore capability | DART-0.1 / 0.2 / 0.5 | ✅ |
| Compute-budget confound was identified and addressed with matched controls | DART-0.2 → 0.3 | ✅ |
| Early transfer can occur for some task pairs | DART-0.3 / 0.5 | 🟡 |
| Small MLP initially dominated replacement search | DART-0.4 / 0.5 | ✅ finding |
| Multi-layer trajectory replacement alone is insufficient | DART-0.6 | ✅ finding |
| Preserving routing dramatically improves capability retention | DART-0.8 | ✅ strong evidence |
| Structured operator can win the DART search | DART-0.9 | ✅ |
| Joint shared primitive discovery is possible | DART-1.0 | ✅ |
| Tiny task-code conditioning solves transfer | DART-1.1 | ❌ |
| Behavioral-signature-based primitive discovery solves transfer | DART-1.2 | ⏳ unvalidated |

---

## 8. Current research scorecard

The long-term goal is larger than compression, so the project should not be judged only by parameter count.

| Research question | Current status |
|---|---|
| Can neural computation be replaced with cheaper computation? | ✅ |
| Can capability be preserved after replacement? | ✅ |
| Can adaptation recover capability? | ✅ |
| Can routing be preserved while replacing computation? | ✅ strong evidence |
| Can structured / non-neural replacements be selected? | ✅ locally |
| Can DART consistently beat ordinary neural controls? | ❌ not established |
| Can the replacement be shown to be an algorithm rather than an approximation? | ❌ not established |
| Can a discovered primitive transfer to an unseen related task? | ❌ not established |
| Can DART automatically discover truly reusable computation? | ❌ open problem |

A simple internal milestone score is therefore:

```text
M1 — cheaper replacement                    ✅
M2 — capability preservation                ✅
M3 — structured reusable computation        🟡
M4 — reliable cross-task reuse              ❌
```

That is **2/4 fully achieved core milestones**. This is a project-management score, not a scientific probability or publication metric.

---

## 9. Problems discovered during implementation

DART has encountered both research problems and software problems. They should not be mixed together.

## Research problems discovered

1. DART could behave like ordinary small-model training
Solved partially with matched scratch, distillation, and DART controls.

2. Extra adaptation created a compute-budget confound
Addressed by compute-matched protocols.

3. Transfer was task-pair dependent
Still unresolved.

4. Candidate search preferred neural approximators
Addressed by structured-only DART candidate pools in later versions.

5. Multi-layer replacement destroyed information flow
Addressed by preserving routing in DART-0.8.

6. Residual networks could hide task-specific relearning
Diagnosed in DART-1.0 and replaced by a tiny task code in DART-1.1.

7. Tiny task code still did not enable transfer
Still unresolved; motivated DART-1.2.

## Engineering problems encountered and fixed

During iterative implementation, several concrete software failures appeared, including:

- CPU/CUDA placement mismatch in dynamically created residual modules.
- `cuda` vs `cuda:0` false-positive device validation.
- Running an old `dart08.py` from the Trash directory instead of the active project file.
- Attention capture / hook handling edge cases.
- Adaptation-loop target handling in a neural control.
- Wrapper-level MAC accounting for conditioned cores.

These were implementation bugs, not scientific results, and were fixed before the affected research runs were treated as valid experiments.

---

## 10. What DART is NOT claiming

DART should **not** currently be described as:

- a proven general-purpose model compression algorithm,
- a proven algorithm-discovery system,
- a proven self-optimizing neural architecture,
- a universally faster replacement for Transformers,
- a system that already learns reusable algorithms across arbitrary tasks.

Those are research goals.

The current evidence supports a more precise statement:

> **DART is an experimental framework demonstrating that learned neural computation can be surgically replaced by cheaper structured computation in a controlled small-scale setting, with adaptation and routing preservation playing important roles. The stronger hypothesis — that the system can discover computation that is genuinely reusable across tasks — remains unproven.**

---

## 11. Current experimental environment

The validated experiments use a small synthetic Transformer setting so that the internal computation can be traced and surgically manipulated.

Typical configuration across later experiments:

```text
d_model      = 32
heads        = 2
d_ff         = 128
depth        = 3
train_size   = 6000
verifier     = 1500
test         = 1500
teacher_steps= 800
CUDA         = used for full runs
```

This small setup is intentional. The purpose is to isolate the mechanism before attempting to scale it.

A positive result on this benchmark would therefore be **evidence for the mechanism**, not evidence that DART automatically works on large frontier models.

---

## 12. Repository organization

A recommended GitHub layout is:

```text
DART/
├── README.md
├── research/
│   ├── DART_research_master_backup.txt
│   ├── THEORY.md
│   └── experiment-notes/
├── dart0/
│   └── ...
├── dart0_1/
│   └── ...
├── dart0_2/
│   └── ...
├── dart0_3/
│   └── ...
├── dart0_4/
│   └── ...
├── dart0_5/
│   └── ...
├── dart0_6/
│   └── ...
├── dart0_7/
│   └── ...
├── dart0_8/
│   └── ...
├── dart0_9/
│   └── ...
├── dart1_0/
│   └── ...
├── dart1_1/
│   └── ...
└── dart1_2/
    └── ...
```

For reproducibility, preserve:

- source code for every version
- command logs
- result JSON files
- theory / hypothesis notes
- failed experiments when scientifically relevant
- Git tags for each experiment

Recommended tags:

```text
dart-0.0
dart-0.1
dart-0.2
dart-0.3
dart-0.4
dart-0.5
dart-0.6
dart-0.7
dart-0.8
dart-0.9
dart-1.0
dart-1.1
dart-1.2
```

---

## 13. Reproducibility principle

Every DART version should answer four questions:

1. **What hypothesis changed?**
2. **What experiment isolates that hypothesis?**
3. **What result supports or falsifies it?**
4. **What is the next experiment if it fails?**

A failed version is not deleted or hidden. It is retained because DART is being developed as a research program, not only as a software product.

---

## 14. Roadmap

## DART-1.2 — Current

Behavioral / relational invariant discovery.

Goal:

> Remove explicit learned task codes and ask whether the reusable structure is visible in the task's observable computational behavior.

Status: **implemented; full research validation pending**.

## DART-1.3+

The exact next version should be determined from DART-1.2 evidence rather than guessed in advance.

Possible directions, depending on the result:

```text
If behavioral signatures work:
    test stronger unseen-task generalization
    increase task diversity
    test primitive composition

If signatures fail:
    inspect the representation of the computation itself
    test token-level / causal subgraph invariants
    separate task structure from state representation

If structured operators transfer:
    test multi-operator libraries and composition

If transfer still fails:
    revisit the definition of “reusable primitive”
    and identify the true invariant across tasks
```

The project should continue to **change the hypothesis when evidence requires it**, rather than repeatedly increasing model size or candidate count.

---

## 15. Research philosophy

DART follows a simple loop:

```text
Hypothesis
   ↓
Implementation
   ↓
Controlled experiment
   ↓
Failure / success signal
   ↓
Identify the exact bottleneck
   ↓
Change one important part
   ↓
New experiment
```

The goal is not to make every version look successful.

The goal is to progressively reduce the space of explanations until we can answer the original question:

> **Can a learning system turn part of its own learned neural computation into a cheaper, reusable computational procedure?**

As of DART-1.1, the answer is:

> **Cheaper replacement: yes. Capability-preserving surgery: yes. Structured replacement: yes in the tested setting. Reliable reusable cross-task computation: not yet.**

That unresolved final question is the main reason the research continues.

---

## License / status

This repository represents an **active research prototype**, not a production-ready library.

Experimental results are specific to the reported benchmark configurations. Claims should be updated whenever stronger controls, larger task sets, or larger models are introduced.
