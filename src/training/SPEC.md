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

- **`game.py`** — `Position` (the state object: `board` · `color` · `fen` · `tensor`) and
  `ChessGame`: 8-plane encoding (6 signed piece planes + castling + en-passant),
  the 133-action `MoveMapper` (64 quiet + 64 captures + 5 castles; promotions resolved at
  execution), `python-chess` legality, termination on mate/stalemate/threefold/insufficient
  material/150-halfmove (adr-0006).
- **`search/`** — the search + self-play package. `_core.py` holds the **SearchCore** — the
  MCTS algorithm math both searches share verbatim (PUCT score, Dirichlet root-noise, and the
  sign-flip backup), numpy-compatible so each strategy calls it with scalars or arrays. The two
  searches keep their own loops + node storage: `vectorized.py` (`VectorizedMCTS` — the
  default; flat numpy pool, one batched net call per iteration, ADR-0007) and `mcts.py`
  (unbatched `MCTS` — legacy ablation path). `heuristic.py` holds the `SunfishSearch` /
  `AlphaBetaSearch` baselines; `selfplay.py` holds `ParallelSelfPlay` (process pool, one
  `BatchManager` inference server for non-vectorized searches, real full games, value target
  `(1-2c)v + 2c·winner·color`). `__init__.py` re-exports the public names so importers are
  unchanged.
- **`network.py`** — `AlphaZeroNet`: ResNet, f128/b4 (candidate B, adr-0009).
- **`training.py`** — `AlphaZeroTraining`: policy log-softmax cross-entropy + value MSE.
- **`evaluation.py`** — `SunfishEvaluation`: material + piece-square evaluation (reads
  `Position.board` directly, returns score from `Position.color` perspective); `ZeroEvaluation`:
  zero-opponent evaluator. PST tables and piece values live in `evaluation_data.py`.
- **`evaluation_data.py`** — PST tables (`pst`) and piece values (`piece_val`) for the
  Sunfish evaluator, extracted from evaluation.py into a stable data module.
- **`system.py`** — `Factory` composition + 250k-move `Buffer` (uniform sampling).
- **`trainer.py`** — orchestration: generate games → buffer → train steps → checkpoint
  (`checkpoint_N.pt`, `model.pt`).

## Boundary

- **State representation**: the game state is a single `Position` value object defined in
  `game.py` — `board` (authoritative: legal moves, FEN, terminal, and winner all derive from
  it, so the hot path never re-parses FEN), `color` (+1 white / −1 black), `fen` (cache), and
  `tensor` (the 8-plane encoded input; may be `None` for heuristic-only positions). Every
  consumer — MCTS, self-play, `Buffer`, `train_step`, evaluation — takes and returns
  `Position`; the former 2/3/4/5-tuples are gone, there is no parallel representation.
- Imports `src.config` / `src.core` only. Nothing here imports `src/infra` — the cloud route
  pulls this module in, never the reverse. The `search/` package sits a level deeper, so its
  modules reach `...config` / `...core` / `..game`; `src.core` never imports `src.training`
  (the ABCs stay representation-agnostic in the leaf).
- **SearchCore (`search/_core.py`)**: the three pieces both MCTS strategies share verbatim —
  `puct_score`, `dirichlet_perturb`, `backup_value` — each numpy-compatible (a scalar for the
  unbatched `MCTS`, an array for `VectorizedMCTS`). These are the shared algorithm; node
  storage (Node tree vs flat pool), net-calling (per-leaf vs batched), and the selection
  driver stay in each strategy — the loops differ by design (ADR-0007).
- `MoveMapper.from_index` **raises** on an action with no legal move; a silent fallback is
  forbidden (it would mask encoding bugs and poison trajectories — the branch's original
  `next(iter(legal_moves))` fallback was removed for this reason).

## Known limitations (draft)

- **MCTS is the unbatched per-leaf form** (`search/mcts.py`, legacy): batch-1 net call per
  leaf, proxied through the queue-based `batch_manager.py` — the "naive batching" adr-0007
  measured 10× worse. The vectorized form (`search/vectorized.py`, `VectorizedMCTS`) is the
  trainer's default and runs in-worker per ADR-0007 (M6: flat-MPS fastest, 2.9× over
  unbatched-CPU) — `batch_manager.py` is now only exercised by the unbatched `MCTS` path. Both
  share the `_core.py` SearchCore (PUCT + Dirichlet + backup); they diverge in node storage
  and net-calling.
- **Sequential data engine**: the Trainer runs generation then training on one static
  network per iteration; adr-0008's concurrent live-weights engine (concurrent MPS
  gen+train, progressive 25→100→250 sims) is not built.
- Buffer samples uniformly at 250k capacity; progressive search budgets are not implemented.
