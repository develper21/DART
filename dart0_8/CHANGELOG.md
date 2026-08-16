# DART-0.8 Changelog

## Structural change
- Stops replacing the entire Transformer block trajectory.
- Preserves every selected block's original attention router and LayerNorms.
- Replaces only the FFN path across the selected trajectory span.
- Uses one shared FF core plus tiny block-specific residual adapters.

## Verification change
- Captures teacher and candidate attention maps.
- Scores candidate-vs-teacher routing agreement.
- Adds attention-output ablation as a causal routing sanity check.
- Keeps related-task transfer testing.

## Research reason
DART-0.6 and DART-0.7 showed that latent/trajectory consistency can rise while
task capability collapses. DART-0.8 isolates information routing as the next
suspected failure point instead of adding more replacement families.
