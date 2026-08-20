# DART-4.6
Symbolic residual planning + verified subgoal decomposition.
DART-4.6 addresses the DART-4.5 failure mode where the heuristic recognized a
target but failed to convert it into an existing primitive composition.
For the current compositional benchmark, symbolic residual plans are compiled
into real primitive references and then exact-verified. The symbolic layer
cannot bypass verification.
