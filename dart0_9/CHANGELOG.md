# DART-0.9 changes

## From DART-0.8

- Kept attention/routing-preserving replacement scaffold.
- Removed MLP from the DART winner pool.
- Added `affine_polynomial` structured candidate.
- Added explicit neural `mlp_control`.
- Added explicit `distilled_mlp_control`.
- DART winner selection now uses only candidates with `eligible=true`.
- Candidate records label mechanism kind: `dart_structured`, `neural_control`, or `distillation_control`.
- Research objective is now a controlled mechanism comparison rather than another generic compression search.
