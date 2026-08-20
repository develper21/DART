# DART-4.0 Theory

The central hypothesis is that DART can discover reusable algorithmic structure instead of selecting from a fixed list of known programs.

A program is represented as a DAG:
- input nodes
- primitive computation nodes
- explicit bindings
- output node

Search grows from shallow graphs to deeper compositions, with exact semantic pruning.

The first milestone is not benchmark accuracy. It is proof that a hidden task can be represented by a small verified graph whose primitives are reusable across tasks.
