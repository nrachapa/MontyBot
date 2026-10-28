---
id: adr-0008
type: architecture-design
title: "ADR-0008: Self-play data engine (concurrent MPS, progressive search, live weights)"
status: draft
parent: goal
---

# Self-play data engine: concurrent MPS, progressive search, live weights

The self-play data engine runs generation and training **concurrently on the MPS GPU** (one process pair; the build measures contention in the first hour and falls back to generation-MPS + training-CPU if combined throughput loses). Generation plays the **live** torch weights (true AlphaZero behavior — no stale data within a stage), feeding a ~250k-move ring buffer (≈0.5 GB) sampled uniformly. The MCTS search budget is **progressive 25 → 100 → 250 sims/move** across the run's stages, repurposing the repo's existing `progressive_trainer` / `get_progressive_configs` scaffolding: search depth is spent where marginal strength is, early games don't need deep search. ONNX is exported once at the end for deployment (ADR-0003/0005/0007), never re-exported between stages.
