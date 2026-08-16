# DART-0.3 scientific question

DART-0.2 showed that post-replacement adaptation can recover performance.
However, DART received extra adaptation steps relative to some controls.

DART-0.3 therefore tests:

    At approximately matched task-training compute,
    does a surgically replaced + adapted model outperform
    a smaller model trained from scratch?

A second question:

    Does a replacement learned on Task A transfer useful computation
    to Task B, or is it merely a Task-A-specific approximation?

Falsifiers:
- Scratch wins reliably at matched compute.
- Distillation + adaptation wins reliably.
- Transfer gives no advantage.
- Results disappear across seeds.

No novelty claim is made until these controls are passed.
