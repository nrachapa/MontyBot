Type: grilling
Status: resolved
Blocked by: 01, 05, 06, 07
Label: wayfinder:grilling

# Self-play data engine design

## Question

Given 07's vectorized-MCTS-on-MPS design and 06's chess `Game` (8 planes, 133 actions, python-chess legality), how should the generation → replay buffer → training loop be shaped?

## Answer (resolved 2026-10-12, HITL — recorded in [ADR-0008](../../../docs/adr/0008-data-engine.md))

1. **Concurrent on MPS**: one generation process + one training process sharing the GPU; build measures contention in the first hour and falls back to generation-MPS + training-CPU if the sum loses.
2. **Live weights**: the vectorized generation loop reads the current torch weights each iteration — true AlphaZero behavior, no stale data.
3. **Progressive search budget 25 → 100 → 250 sims/move** across stages (repurposes the existing `progressive_trainer` / `get_progressive_configs` scaffolding).
4. **Buffer**: ~250k-move ring (~0.5 GB at chess scale: 8×8×8 + 133 + 1 float32 per move), uniform sampling; the existing `ReplayBuffer` shape carries over.
5. **No re-export cadence**: generation is torch-on-MPS (ADR-0007); ONNX is exported once at the end for terminal play + evaluation (ADR-0003/0005).
6. **python-chess cost management** (caching/deferring legal-move scans in the numpy loop) is implementation work inside the build, per 07 — not a further decision.

**Grads from the map's fog:** the evaluation-round *cadence* (ADR-0005's "1-2 rounds/day, at stage boundaries") is implementable now that the engine shape is fixed; the build takes it from ADR-0005, no new ticket.
