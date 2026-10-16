# Coding standards

Rules for the reviewer agent. Mechanical rules live in `ruff.toml` (enforced by
pre-commit and CI — don't re-adjudicate what the linter already catches).
What remains here are the judgement calls.

## Prototypes self-check before they spend machine time

A prototype that trains, plays, or benchmarks must verify its own pipeline
before a full run:

1. **Finite outputs**: assert `torch.isfinite(loss)` (and net outputs) after the
   first few steps. A `nan` that survives 12 minutes of training is a silent
   invalid result, not a slow one.
2. **Smoke every phase, not just the first.** A one-game / one-batch /
   one-call smoke of *each* distinct phase (generation, training, evaluation)
   before launching the full budget.
3. **Timing is median of ≥ 3** runs over varied positions, not one run from
   the start position. Same net measured 490 ms and 800 ms/move across two
   single-run measurements on MPS; a phantom 60% cost difference is not a
   decision-grade number.
4. **One honest caveat per number**: state what the measurement cannot tell
   you (e.g. "tiebreak has no statistical power at 12 games").

## Numbers are reported with their noise

When presenting measured costs or win-rates, carry the variance with them
(min–max or ±σ across repeats). A single noisy measurement presented as exact
is how a wrong decision gets made with a straight face.

## Error handling and validation survive refactors

When code moves or gets merged, its error handling and validation move with it.
A refactor that silently drops a check is a behaviour change, not a cleanup.
