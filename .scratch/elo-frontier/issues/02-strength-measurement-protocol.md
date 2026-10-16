Type: grilling
Status: resolved
Blocked by: (none)
Label: wayfinder:grilling

# Strength measurement protocol

## Question

How do we make "win-rate against the reference policy" a trustworthy, cheap strength signal? Settles: games-per-measurement and when a result is significant enough to count (binomial thresholds), how draws are handled, how the reference policy is chosen and re-anchored as checkpoints improve, and how measurement time is reserved out of the 24h budget.

## Answer (resolved 2026-10-12, HITL)

**Protocol (owner-approved, recorded in [ADR-0005](../../../docs/adr/0005-strength-measurement.md)):**

1. **Promotion gate — size only**: win% − loss% ≥ **10 points** over the round. No significance test — the owner explicitly rejected the binomial p < 0.05 gate; do not reintroduce it without reopening ADR-0005.
2. **Draws**: counted as ½ in the cost-curve number, (W + ½D) / games; the gate uses win%/loss% of non-drawn games only, draws reported alongside.
3. **Reference**: re-anchored to the promoted checkpoint on every promotion, so each round measures *marginal* gain. A checkpoint that fails the gate stays the reference.
4. **Budget**: evaluation runs on the **ORT/CPU** path (`ort_inference.py`, the terminal-play backend) so MPS stays 100% for training; ~1-2 rounds/day, at stage boundaries and after any checkpoint that looks like a jump.
5. Fixed MCTS sim budget per side during evaluation (same for reference and challenger) so rounds are comparable; rounds use color alternation.

**Implementation note for the build:** `evaluate.py` + `Config.eval_games/eval_frequency` are the existing scaffolding; the build adapts them to this protocol (gate + re-anchor + ORT backend) — that is implementation work, not part of this decision.
