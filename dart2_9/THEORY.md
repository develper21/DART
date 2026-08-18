# DART-2.9 Theory

DART-2.7/2.8 suggest a compact causal primitive can be localized and frozen, but scalar primitive reparameterization alone underperforms on unseen transfer. DART-2.9 therefore tests whether task-specific computation is better modeled as a small transformation of the primitive's input and/or output interface.

The primitive is frozen. The target task receives only a bounded two-scalar interface transform per enabled side. This isolates interface expressiveness from primitive retraining.

A successful result requires: frozen-primitive transfer, very small interface parameter count, positive target gain over zero-shot, superiority to primitive permutation/random controls, and rejection of the unrelated contrast task.
