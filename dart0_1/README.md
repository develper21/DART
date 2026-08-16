# DART-0.1

Controlled experiment comparing:
- Original teacher
- Scratch-small
- Distill-small
- DART (surgery, no adaptation)
- DART+Adaptation

Metrics:
- task accuracy/loss
- total parameters
- target FFN parameters
- target FFN MACs/token
- CUDA latency
- peak CUDA memory
- training/replacement time

Run on CUDA:
```bash
python3 dart01.py
```

Fast smoke run:
```bash
python3 dart01.py --baseline-steps 100 --scratch-steps 100 --replacement-steps 80 --adaptation-steps 80 --train-size 2000 --test-size 500 --trace-batches 10 --latency-iters 20
```

The strongest DART signal is NOT just high accuracy. We want DART+Adaptation to retain or recover capability while retaining the cheaper target computation, and to beat ordinary distillation / scratch-small controls on the capability-vs-compute frontier.
