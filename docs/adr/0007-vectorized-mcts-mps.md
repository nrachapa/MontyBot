---
id: adr-0007
type: architecture-design
title: "ADR-0007: Vectorized MCTS generation"
status: active
parent: goal
---

# Vectorized MCTS generation

Self-play generation uses a fully vectorized MCTS: a flat numpy node pool (one array per node field; the per-node `chess.Board`, FEN, and (8,8,8) tensor live in object slots), batched PUCT selection, and one batched net call per iteration over the top-32 frontier leaves. We rejected (a) the original per-leaf Python tree — unbatched measured 190 ms/move CPU vs 216 ms MPS at 32 sims (ticket 09) — and (b) naive leaf-batching over that per-leaf tree (32 leaves/call) — 2–3× *slower* than unbatched, because per-leaf Python cost is backend-independent and dwarfs the ~1 ms batched net call. The flat pool is what fixes (b): the hot loop (PUCT + net + backup) touches numpy only, and python-chess cost (legality mask, FEN, encode) is paid exactly once per *new* position, at expansion — the caching/deferral this ADR requires of python-chess (ADR-0006 keeps it the legality source).

**Measured 2026-10-30 (ticket 10, `.scratch/elo-frontier/proto_flat_mcts.py`, f128/b4 net, torch 2.2.2, the M6; random weights — loop cost is weight-independent):**

| work | flat (batch 32) | unbatched | flat / unbatched |
|---|---|---|---|
| 32 leaves, CPU | 4.51 ms/leaf | 5.12 ms/leaf | 0.88× |
| ~800 leaves, CPU | 3.98 ms/leaf (3.13 s/move) | 5.54 ms/leaf (4.43 s/move) | **0.71× (1.4× faster)** |
| 32 leaves, MPS | 2.29 ms/leaf | 5.35 ms/leaf | 0.43× |
| ~800 leaves, MPS | 2.32 ms/leaf (1.83 s/move) | 6.70 ms/leaf (5.36 s/move) | **0.34× (2.9× faster)** |

**Acceptance, 2026-10-30, M6, 1.21M-param production net, sims=250, median of 3 full games (`.scratch/elo-frontier/bench_prod_m6.py`):** flat-MPS **45 s/game = 80.8 games/h** — 14.7× the unbatched-CPU baseline (5.5 games/h), 1.7× flat-CPU. Batch amortization grows with net size and leaves/iteration, so the production net shows far more than the random-weight proto's 2.9×. Acceptance bar (≥ proto parity on MPS at equal sims) met; `gen_device=mps` is the local default, `cuda` for cloud jobs.

The batch amortization materializes only on MPS (a 32-leaf call ≈ 2× a 1-leaf call); on CPU a 32-leaf call costs ≈32× a 1-leaf call (the 1.2 M-param net's matmul is too small to amortize), so the flat loop's CPU win is the per-leaf Python elimination alone. Consequences, all implemented in ticket 10:

- **`simulations` means leaf visits** in both strategies: a flat search at `simulations=N` runs `ceil(N / 32)` batched iterations, so the config knob keeps one meaning across `MCTS` and `VectorizedMCTS`.
- **MPS is now the measured-fastest local generation backend** (1.83 s/move at ~800 leaves vs 4.43 s unbatched CPU) — this supersedes ticket 09's "keep generation on CPU" conclusion, which measured the *unbatched* loop only (1.14× MPS/CPU gap). `gen_device` stays a config knob with default `cpu` for portability (a literal `mps` default would break the CUDA-only SageMaker container); M6 local runs set `gen_device=mps`, cloud jobs `cuda`.
- Board/FEN/tensor consistency is a node invariant (one position per node; verified by `.scratch/elo-frontier/verify_flat.py`), and a passed root whose FEN no longer matches resets the pool.
