Type: research
Status: resolved
Blocked by: (none)
Label: wayfinder:research

# Generation throughput on the M6

## Question

What games/hour does the current self-play pipeline actually produce on this machine (M6, 12 cores, MPS, with and without the turbo thread cap) at the current network size — and is generation, or training, the binding constraint against the 24h budget?

## Answer (resolved 2026-10-12, measured as-is, no redesign)

Environment: repo `.venv` (Python 3.10, torch 1.13.1, MPS available). Benchmark script: `graphify-out/bench_gen.py` (kept as a primary source).

| Phase | Config | Throughput |
|---|---|---|
| MCTS move, MPS | f256, b10, sims 800 (default GPU config) | **1.94 s/move** |
| Full game, MPS | same | **9.0 s/game** (~8 moves/game) → ≈ **400 games/h** single-process, ≈ 3,200 moves/h |
| MCTS move, CPU | f64, b2, sims 25 (fast config) | **0.012 s/move** |
| Full game, CPU | same | **0.07 s/game** → ≈ **420,000 games/h** with 8 workers |
| Train step, MPS | f256, b10, batch 64 | **0.035 s/step** |
| Train step, CPU | f64, b2, batch 32 | **0.012 s/step** |

**Verdict:** for the game this repo actually implements, neither generation nor training is the binding constraint — training capacity (~90k samples/h on MPS) dwarfs generation output (~3.2k moves/h); the pipeline is massively underutilized at current scale. Compute is not the bottleneck; **game complexity is** (see the finding below).

**Key findings:**

1. **⚠️ The "chess" game is 3×3 tic-tac-toe.** `training/game.py` implements a 3×3 board (9 cells, row/col/diagonal wins). Nothing in the repo implements real chess (8×8, ~30 moves/game, 189–191 actions). All throughput numbers above are for tic-tac-toe. Real chess is roughly 5–8× longer games at ~7–10× the board/cell cost per move → a crude extrapolation puts MPS self-play at **~7–8 min/game ≈ 200–250 games per 24h** at the current config.
2. **Bug: `BatchedSelfPlay` never moves the network to its device** — the MPS path crashes at first inference (`MPSFloatType vs FloatType`). The CPU path only works because everything defaults to CPU. (Found while benchmarking; fix belongs to implementation work.)
3. **MCTS is a pure-Python unbatched loop** (`mcts.run`): even on MPS, each simulation does a single batch-1 net call. The `batch_evaluation` flag exists but nothing batches. This is the largest per-move overhead and the main lever if real-chess generation must speed up.

Asset: `graphify-out/bench_gen.py`
