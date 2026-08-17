# DART-1.5 Theory

## Failure being targeted

DART-1.4 selected a diagonal factorized primitive with `theta = [0,0,0,0]` and no measurable theta adaptation. This means the parameterized factorization was not actually being used. Its causal score was also much higher than its relational agreement, suggesting that the causal verifier was insufficiently discriminative.

## DART-1.5 hypothesis

A reusable algorithmic primitive should have **necessary parameters**. If the mechanism is truly factorized, then task identity must alter the primitive's behavior through a small explicit parameter vector.

For `C(x, theta)` we therefore require:

- `||theta|| > epsilon`
- `C(x, theta + delta e_k) != C(x, theta - delta e_k)` for meaningful k
- theta estimates remain stable across data splits
- the primitive retains meaningful source-task capability
- the relational response agrees with the teacher

## Why hard gates?

A soft score can be dominated by cheapness or by a weak proxy metric. DART-1.5 instead treats scientific validity as sequential eligibility tests. Complexity is used only after a candidate has demonstrated the properties we actually care about.

## Expected outcomes

### Strong positive

The same frozen `C` supports multiple source tasks with different stable theta vectors and a small theta fit transfers the mechanism to `sub`.

### Diagnostic failure A

Theta remains near zero or has no causal effect: the factorization is not identifiable.

### Diagnostic failure B

Theta is non-trivial and causal, but source capability remains poor: the primitive family is inadequate.

### Diagnostic failure C

Source tasks are well explained and theta is meaningful, but `sub` fails: the discovered mechanism is task-specific rather than invariant.
