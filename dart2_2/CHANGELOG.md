# DART-2.2 Changelog

## 2.2.0

- Replaced scalar task-theta configuration with a structured task-operator grammar.
- Added identity, scale, negate, difference, product, and mix operators.
- Added operator-effect necessity measurement.
- Added full operator-permutation matrix and specificity score.
- Preserved rule-level causal intervention testing.
- Preserved random-graph and separate-graph controls.
- Target transfer freezes the shared graph and fits only the task operator.
- Fixed operator fitting so target adaptation uses the actual frozen shared graph rather than a fresh graph instance.
