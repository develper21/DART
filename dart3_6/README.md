# DART-3.6

Semantic task-law validation + exact-law extrapolation.

DART-3.6 adds an explicit semantic law oracle to prevent a structurally consistent but semantically wrong task law from being accepted. The law is compiled deterministically into a small program around a frozen shared primitive. Regimes A-D are discovery/validation regimes; E remains untouched for final extrapolation.
