# DART-0 Next Experimental Step

The first candidate was rejected.

Why:
- Replacement was dramatically smaller (812 vs 8,352 FFN parameters).
- Hidden-state counterfactual error was not low enough.
- End-task accuracy dropped by 4 percentage points.

Do NOT relax the acceptance threshold just to get a success.

Next modification:
1. Train multiple replacement candidates with different bottlenecks.
2. Measure the accuracy-vs-cost frontier.
3. Add a local repair phase after surgery.
4. Re-test after repair.
5. Compare against:
   - original model;
   - ordinary distillation;
   - same-size small FFN trained from scratch.
6. Only call DART promising if it beats these controls.

Core scientific question:
Does "learn replacement -> surgically replace -> adapt" provide a better compute/quality frontier than ordinary distillation or smaller-model training?
