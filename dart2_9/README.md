# DART-2.9

## Causal Interface Reparameterization around a Frozen Primitive

DART-2.7/2.8 produced a stable localized causal primitive, but tiny scalar reparameterization did not transfer strongly to an unseen task. DART-2.9 tests whether task variation belongs at the primitive input/output interfaces rather than inside the frozen primitive.

### Target function

`y_t = I_out,t(P*(I_in,t(x)))`

`P*` is frozen during target transfer. Only tiny bounded input/output interface parameters are trained.

### Transfer controls

- zero-shot frozen primitive
- input-only interface
- output-only interface
- input+output interface
- primitive permutation control
- random primitive control
- MLP control

The live progress bar remains enabled.
