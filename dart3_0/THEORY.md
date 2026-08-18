# DART-3.0 Theory

DART-2.8 showed that tiny scalar reparameterization of a frozen primitive was insufficient. DART-2.9 showed that small input/output affine interfaces can improve transfer but do not yet solve it.

DART-3.0 therefore tests a short structured task program around the frozen causal primitive:

    x -> TaskProgram -> P* -> TaskProgram/Decode -> y

The program grammar is intentionally small and complexity-penalized. Target transfer freezes the discovered primitive.
