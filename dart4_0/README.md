# DART-4.0

Open primitive + compositional program-graph discovery.

DART-3.9 demonstrated exact verified programs across several bounded binary tasks. DART-4.0 changes the search object itself:

    fixed flat program
        -> open primitive library
        -> program graph
        -> variable-arity tasks
        -> nested composition
        -> exact verification

The semantic verifier remains the acceptance authority. A graph is accepted only if it exactly matches the task oracle on symbolic probes, randomized probes, A-F regimes, and far-OOD probes.

The neural model is a diagnostic layer only.
