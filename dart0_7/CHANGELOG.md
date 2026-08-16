# DART-0.7 changelog

## From DART-0.6

- Changed the replacement equation from `h' = C(h)` to `h' = C(h) + R_i(h)`.
- The computational core `C` is shared across all trajectory positions.
- Added a tiny step-specific residual adapter for each trajectory transition.
- Added residual complexity accounting and residual-fraction reporting.
- Core selection is evaluated jointly with residual correction rather than asking one operator to reproduce the entire Transformer span.
- Added an explicit shared-MLP control with the same residual interface.
- Added post-surgery residual/core adaptation without rebuilding the original Transformer span.
