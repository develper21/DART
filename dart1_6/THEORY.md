# DART-1.6 Theory

### Hypothesis

A useful reusable computational primitive may be **compositional** rather than a single elementary structured map.

DART-1.5 showed that an explicit low-dimensional theta can be necessary and causally effective, but the shared primitive may still be too weak to express source and target computations.

DART-1.6 therefore searches over shallow compositions of structured operators. The intended separation is:

- shared mechanism = the ordered primitive composition;
- task-specific information = small explicit theta values;
- transfer = freeze the shared mechanism and refit only theta.

### Falsification

The hypothesis is weakened if compositions do not improve source capability over single primitives, theta becomes irrelevant, transfer does not improve, or the composed mechanism behaves like an unconstrained neural approximation.
