# DART-1.7 Theory — Typed Intermediate Representation Discovery

## Hypothesis
DART-1.6 showed shallow composition can be discovered, but composing operators directly in the raw hidden-state space remained too weak for reliable transfer. DART-1.7 tests whether a reusable computational law becomes visible in a structured intermediate representation.

The DART primitive is factorized as:

`y = D(T(E(x), theta))`

where `E` is a structured extractor, `T` is a structured transformation, `D` is a structured recombination map, and `theta` is a tiny explicit task parameter.

## New evidence target
A useful representation should be stable across the related source tasks and less aligned with an unrelated contrast task. The experiment therefore reports interface invariance, theta necessity/stability, source capability, related-task transfer, contrast behavior, and a matched MLP control.

## Falsification
DART-1.7 fails its hypothesis if the typed representation gives no source-task benefit, has poor cross-task invariance, theta becomes unnecessary, or frozen transfer to `sub` remains no better than the neural control.
