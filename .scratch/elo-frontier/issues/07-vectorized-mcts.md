Type: grilling
Status: resolved
Blocked by: 05
Label: wayfinder:grilling

# Vectorized MCTS generation design (the MPS lever)

## Question

From 01: MPS raw net compute scales 10× with batch size, but naive 32-leaf batching was 2-3× *slower* at 3×3 scale. Decide the generation architecture at **real chess scale** (06's 8-plane representation) before 03 designs the data engine around it.

## Answer (resolved 2026-10-12, HITL + prototype)

Prototype: `.scratch/elo-frontier/proto_chess_mcts.py` (python-chess game, 8 planes, 133 actions, f32/b1 net — loop architecture, not FLOPs, is what was measured).

**Measured (solo process, ~80-move games):**

| Variant | ms/move | games/h |
|---|---|---|
| MPS, unbatched, 100 sims | 141 | ~319 |
| MPS, batched-32, 100 sims | 1520 | ~30 |
| CPU, unbatched, 100 sims | 83 | ~543 |
| CPU, batched-32, 100 sims | 1509 | ~30 |

**cProfile of the batched run:** 2.58 s, of which ~1 ms is torch; the rest is per-leaf Python — 1.8M `piece_at`, 553k legal-move scans, 72k board copy+push. The batched-32 variant is *worse* because per-leaf Python cost (~15 ms/sim in this naive form) dwarfs the batched net call (~1 ms for 32 leaves). Also note batched-32 costs ~1520 ms on MPS **and** CPU — the Python floor is backend-independent, confirming 01's hypothesis: **the loop, not the device, is the bottleneck.**

**Decision (owner-confirmed):**
1. **Fully vectorized MCTS**: flat numpy tree, batched PUCT selection/backup, one batched net call per iteration, batched board-plane conversion. This removes the per-leaf Python floor that made naive batching lose; the 3×3 and chess-scale data both point the same way.
2. **Generation backend: MPS-torch** for the vectorized loop. Rationale: unbatched CPU is only 1.7× faster (83 vs 141 ms/move) with a tiny net — the gap *widens* as the net grows (ticket 04), and MPS batched net compute scales ~10× with batch size (01). ORT is **not** the generation backend, but stays where ADR-0005 already put it: the evaluation/measurement path and terminal play (CPU, $0, no GPU contention).
3. python-chess stays the **legality** source (ADR-0006); the vectorized design must handle its per-position cost (caching/deferring legal-move scans) — that's 03/build work, not a new decision.

**Consequence for the cost curve:** expect generation to move from the ~319 games/h unbatched floor toward the MPS batched-compute ceiling once vectorized; the *validated* chess-scale games/h number is a build-phase measurement, and 03 designs the buffer/throughput around whatever it is.
