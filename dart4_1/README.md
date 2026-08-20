# DART-4.1

Blind primitive discovery + reusable primitive library.

The defining change from DART-4.0 is reuse-before-invention: DART first searches its verified primitive library, and only if no verified existing structure solves the holdout does it induce a new primitive graph.

Every inserted primitive must pass exact symbolic, randomized, A-F, and far-OOD verification. Provenance and use counts are persisted so later tasks can reuse earlier discoveries.
