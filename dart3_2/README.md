# DART-3.2

Causal counterfactual task-program synthesis around a frozen shared primitive.

DART-3.2 extends DART-3.1 by evaluating not only whether a task program improves accuracy, but whether removing each program step produces a teacher-aligned counterfactual change on the same trained model state.

Key properties:
- frozen shared primitive during target adaptation
- explicit short task-program grammar
- same-state program necessity
- teacher counterfactual fidelity
- target train/validation/test split
- untouched target test for final reporting
- program permutation and random-program controls
- no "repair" label in the terminal progress output
