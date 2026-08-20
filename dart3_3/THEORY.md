# DART-3.3 Theory

Core hypothesis:

    A true task-level program should remain behaviorally and causally stable
    across independent distributions of the same task.

Target protocol:

    A = adaptation distribution
    B = independent validation distribution
    C = untouched final test distribution

A program is selected using A+B consistency, not C. The final evaluation is on C.

The program score combines minimum performance across A/B, minimum causal necessity, minimum teacher-counterfactual fidelity, cross-distribution invariance, and a length penalty.
