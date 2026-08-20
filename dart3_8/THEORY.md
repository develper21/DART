# DART-3.8 Theory

The core bottleneck after DART-3.7 is the Law -> Program compiler.

DART-3.8 treats compilation as a verification problem, not a heuristic mapping.

For a candidate program P:

    verify P(x) == oracle_task(x)

over:
- deterministic symbolic probes,
- randomized probes,
- regimes A/B/C/D/E.

Only a fully verified P may be integrated with the frozen neural primitive.

Primary metric: exact semantic agreement.
Neural accuracy remains a secondary diagnostic.
