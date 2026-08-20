# DART-3.9

Open multi-holdout verified task-law/program generalization.

DART-3.9 rotates the holdout task instead of validating only subtraction. Each holdout gets:
- invariant-law discovery across A-D
- exact candidate program synthesis
- symbolic and randomized proof over A-F
- shortest verified-program selection
- cross-seed program stability
- explicit anomaly reporting

A run is allowed to report `NO_VERIFIED_PROGRAM`; the framework never silently upgrades a failed holdout into success.
