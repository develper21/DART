# DART-1.1 Changelog

## From DART-1.0
- Removed per-task residual networks from the shared-primitive mechanism.
- Added a tiny task-conditioning vector (`task_dim`, default 4).
- Added a shared task-code conditioner around the structured core.
- Frozen shared core + conditioner during unseen-task transfer.
- Adapt only the holdout task code.
- Report centroid-code zero-shot and zero-code zero-shot separately.
- Keep MLP as a neural control only; it cannot become the DART winner.
- Add explicit task-code parameter accounting.
