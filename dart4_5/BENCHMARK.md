# DART-4.5 benchmark

Recommended full run:
--seeds 1 2
--all-tasks add sub mul absdiff max min sum3 pairdiff3 compose
--holdout-tasks sub sum3 pairdiff3 absdiff max min
--contrast-tasks max
--max-graph-depth 3
--max-graph-nodes 10
--max-plan-depth 4
--max-search-states 192
--max-reuse-depth 4
--top-k-retrieval 8
--mixed-prefix-top-k 4
--lookahead-top-k 4
--lookahead-bindings 8
--goal-backslide-tolerance 0.05
--verification-eps 1e-6
--device cuda
