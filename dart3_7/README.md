# DART-3.7

Semantic execution equivalence + exact algorithm validation.

DART-3.6 established semantic task-law validation. DART-3.7 separates exact algorithm semantics from the learned neural benchmark:

1. infer/validate the task law,
2. compile it deterministically,
3. execute the compiled program directly on symbolic probe inputs,
4. verify exact agreement with the task oracle,
5. run a frozen-primitive neural diagnostic separately.

Primary evidence is exact semantic execution, not teacher accuracy. The final regime E remains untouched for neural extrapolation.
