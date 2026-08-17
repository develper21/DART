# DART-1.4 Changelog

## 1.4.0

- Replaced single structured primitive with a factorized shared basis.
- Added explicit low-dimensional task coefficients (`theta`).
- Removed task ID / learned task embedding / conditioner / target residual from DART.
- Added joint multi-task primitive discovery.
- Added source-theta stability measurement using split observations.
- Added frozen shared-basis transfer to a related holdout.
- Added centroid-theta zero-shot baseline.
- Added theta-only holdout adaptation.
- Added matched routing-preserving MLP control with control-only trainable parameters.
- Added related-vs-contrast task evaluation.
- Candidate scoring now uses meta-task verifier data rather than the training split.
