# DART-4.2 Theory

The central hypothesis is:

    verified primitive knowledge should be reusable.

A source-task library is built before holdout testing. Holdouts cannot mutate
that library before their final selection. Candidate retrieval uses arity and
behavioral/law signatures. Existing primitives are tried first, compositions
second, and new primitive discovery is a fallback.

The decisive result is not only exactness; it is a non-zero, verified reuse rate
and at least one verified multi-primitive composition on a held-out task.
