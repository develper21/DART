# DART-4.2

Persistent primitive retrieval + compositional reuse.

DART-4.1 demonstrated exact primitive discovery and storage but achieved zero
primitive reuse. DART-4.2 changes the evaluation protocol so non-holdout source
tasks first build a persistent primitive library. The holdout library is then
frozen and the solver follows:

    retrieve -> direct reuse -> composition reuse -> new primitive fallback

Every selected primitive/program must pass the exact semantic verification gate
before it is accepted.
