# DART-2.3 Theory

DART-2.2 showed that a single task operator attached after a shared rule graph did not close the gap to separately specialized graphs. DART-2.3 therefore treats task-specific computation as potentially distributed across the shared computation.

The tested factorization is:

`F_t = G3 o A_t,3 o G2 o A_t,2 o G1 o A_t,1`

with a strict cap on the number of active adapters.

The central hypothesis is that most computation is invariant while a small, causally necessary set of task-specific sites explains the remaining algorithmic variation.
