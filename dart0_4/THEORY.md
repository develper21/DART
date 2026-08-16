# DART-0.4 hypothesis

The current hypothesis is narrower than "DART invents algorithms":

> A neural model may contain internal computations whose behavioral response
> can be represented by a cheaper operator. If that operator is validated on
> interventions and the remaining network can adapt after surgery, repeated
> replacement can produce a self-changing computational graph.

DART-0.4 adds two tests not present in DART-0.3:

1. Intervention response:
   The candidate must reproduce how the source computation changes under
   perturbations, not merely imitate a finite set of source outputs.

2. Operator-family selection:
   The candidate is selected from several computational forms with an
   explicit complexity penalty.

The result is still falsifiable. A small MLP that consistently wins and no
cross-task transfer occurs would mean DART has not yet found a novel mechanism.
