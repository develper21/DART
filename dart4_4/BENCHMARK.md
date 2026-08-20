# DART-4.4 Benchmark

Primary run:
- seeds 1,2
- all tasks: add sub mul absdiff max min sum3 pairdiff3 compose
- holdouts: sub sum3 pairdiff3 absdiff max min
- max plan depth 4
- max search states 128
- max graph depth 3
- max graph nodes 10

Required:
- 12/12 exact verified
- source library complete
- zero anomalies
- at least one hierarchical depth >= 3 plan or a logged search exhaustion
- exact reference attribution
- far-OOD verification
