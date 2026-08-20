# DART-3.8

Verified law-to-program compilation.

DART-3.7 showed that a correct semantic law could still compile into an incorrect executable program. DART-3.8 changes compilation into a proof-gated search:

1. infer the task law,
2. enumerate exact candidate programs for that law,
3. test every candidate against the exact task oracle on symbolic and randomized probes across A-E,
4. accept only programs with perfect agreement,
5. choose the shortest verified program,
6. use the verified program in the frozen-primitive neural diagnostic.

If no candidate is semantically exact, DART-3.8 intentionally stops instead of producing a misleading result.
