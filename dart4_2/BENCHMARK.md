# DART-4.2 Benchmark

Recommended primary benchmark:

- all tasks: add sub mul absdiff max min sum3 pairdiff3 compose
- holdouts: sub sum3 pairdiff3 absdiff max min
- source library: all non-holdout tasks
- seeds: 1 2

Interpretation:

- `direct_reuse`: target solved by an existing source primitive.
- `composition_reuse`: target solved by multiple existing source primitives.
- `new_primitive`: retrieval/composition failed, so a new verified primitive was discovered.

A successful DART-4.2 run should have:

1. all holdouts exact,
2. reuse_rate > 0,
3. at least one composition_reuse,
4. no hidden semantic anomalies,
5. provenance clean.
