# DART-4.4

Long-horizon algorithm synthesis.

Builds directly on DART-4.3's verified primitive references and hierarchical
reuse. The main addition is a best-first planner that searches reference
sequences up to configurable depth, memoizes verified graphs, and records
search diagnostics.

A planner proposal is never accepted without the exact semantic verifier.
The source primitive library is frozen before holdout selection.
