# DART-1.2 Changelog

## From DART-1.1
- Removed trainable task code as the primary task representation.
- Removed large per-task residual networks entirely.
- Added deterministic `BehavioralSignature` extraction from fixed probe behavior.
- Added shared `BehavioralConditioner` that maps the signature to core modulation.
- Holdout transfer now uses the frozen shared primitive with the holdout behavior signature.
- Added a separate condition-only adaptation experiment that updates only the shared conditioner on the holdout task.
- Kept matched MLP control for comparison.

## Smoke test
The end-to-end pipeline (meta-teacher training, structured search, meta adaptation, frozen behavioral transfer, conditioner-only adaptation, and MLP control) completed successfully on CPU.
