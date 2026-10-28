---
id: src-training
type: module-design
status: draft
title: "src/training: self-play training pipeline"
parent: src
depends-on: [src]
references: [adr-0001, adr-0006, adr-0007, adr-0008, adr-0009]
---

## Responsibility

The whole AlphaZero loop:

- **`game.py`** — `ChessGame`: 8-plane encoding (6 signed piece planes + castling + en-passant),
  the 133-action `MoveMapper` (64 quiet + 64 captures + 5 castles; promotions resolved at
  execution), `python-chess` legality, termination on mate/stalemate/threefold/insufficient
  material/150-halfmove (adr-0006).
- **`search.py`** — MCTS with PUCT + Dirichlet noise and subtree reuse; `ParallelSelfPlay`
  (process pool, one `BatchManager` inference server, real full games, value target
  `(1-2c)v + 2c·winner·color`); `SunfishSearch`/`AlphaBetaSearch` heuristics as baselines.
- **`network.py`** — `AlphaZeroNet`: ResNet, f128/b4 (candidate B, adr-0009).
- **`training.py`** — `AlphaZeroTraining`: policy log-softmax cross-entropy + value MSE.
- **`system.py`** — `Factory` composition + 250k-move `Buffer` (uniform sampling).
- **`trainer.py`** — orchestration: generate games → buffer → train steps → checkpoint
  (`checkpoint_N.pt`, `model.pt`).

## Boundary

- Imports `src.config` / `src.core` only. Nothing here imports `src/infra` — the cloud route
  pulls this module in, never the reverse.
- `MoveMapper.from_index` **raises** on an action with no legal move; a silent fallback is
  forbidden (it would mask encoding bugs and poison trajectories — the branch's original
  `next(iter(legal_moves))` fallback was removed for this reason).

## Known limitations (draft)

- **MCTS is the unbatched per-leaf form**: batch-1 net call per leaf, proxied through the
  queue-based `batch_manager.py` — the "naive batching" adr-0007 measured 10× worse. The
  vectorized design (flat numpy tree, batched PUCT, one batched call per iteration) exists
  only in `.scratch/elo-frontier/proto_chess_mcts.py` until ticket 10 lands.
- **Sequential data engine**: the Trainer runs generation then training on one static
  network per iteration; adr-0008's concurrent live-weights engine (concurrent MPS
  gen+train, progressive 25→100→250 sims) is not built.
- Buffer samples uniformly at 250k capacity; progressive search budgets are not implemented.
