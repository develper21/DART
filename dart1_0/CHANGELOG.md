# DART-1.0 Changelog

- Changed discovery from single-task/operator fitting to joint multi-task shared-core fitting.
- The same structured core is shared across all meta-training task models and all replaced blocks.
- Added leave-one-task-out evaluation.
- Added true frozen-core zero-shot transfer.
- Added target-only residual adaptation after the frozen-core test.
- Added matched routing-preserving MLP control on the unseen task.
- Added explicit transfer metrics against teacher and MLP control.
- Preserved routing-aware scoring from DART-0.9.
