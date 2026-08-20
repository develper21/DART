# DART-4.4 Theory

DART-4.3 demonstrated explicit primitive references and shallow hierarchical
reuse. DART-4.4 extends this to long-horizon planning.

The planner optimizes, in order:
1. exact semantic correctness,
2. fewer references,
3. smaller depth,
4. search efficiency.

Only exact-verified plans can become final algorithm certificates.
