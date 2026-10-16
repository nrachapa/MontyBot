Type: map
Label: wayfinder:map

# Effort: strongest local self-play chess model

## Destination

The strongest model a day-scale local compute budget can buy: a final checkpoint playable from the terminal (UCI bridge), with the cost curve (win-rate vs. compute spent) documented so the next effort knows how far more time would carry it.

## Notes

- **Working branch: `alphazero`** (ADR-0010) — real chess already implemented in `src/` (python-chess game, 104-plane/4672-action encoding, unbatched MCTS, parallel self-play, f128/b4 ResNet, Sunfish/alpha-beta heuristics, click CLI, SageMaker/S3 cloud path). Written for an Intel MacBook Pro 2019 (CPU era); we are now on an Apple M6 with Metal. **`main` is frozen legacy** (3×3 tic-tac-toe `training/`) — do not work there.
- Domain: MontyBot, an AlphaZero-style chess engine. Read `GLOSSARY.md` at repo root and the ADRs in `docs/adr/` before every session; use their vocabulary.
- Skills to consult: `grilling`, `domain-modeling`, `research`, `prototype`, `codebase-design` (for generation-pipeline shape).
- Compute envelope (ADR-0004, reaffirmed in ADR-0010): Apple M6, 12 cores, 32 GB, MPS GPU. Local is the default; cloud remains an optional override. Runs must be background-friendly by default with a turbo flag, ≤24h, checkpointed.
- Knowledge graph: `graphify-out/graph.json` — query it for architecture questions instead of re-reading code. (Regenerated for the `alphazero` branch, 2026-10-15.)
- One ticket resolved per session (research tickets may resolve several).

## Decisions so far

Authored 2026-10-11/12 while (unknowingly) on `main`; carried over on 2026-10-15 and reaffirmed against the `alphazero` branch in ADR-0010. Where branch code diverges, the ADR is the design of record and the branch code is the object of the fix (tickets 08–12).

- **Training data (01-generation-throughput, resolved):** pure self-play only, no external/supervised data. [ADR-0001](../../docs/adr/0001-pure-self-play-training.md)
- **Strength measurement (02-strength-measurement-protocol, resolved):** win% − loss% ≥ 10 points vs a **fixed reference policy** that re-anchors on each promotion; no absolute Elo; evaluation on ORT/CPU so MPS stays free for training. [ADR-0002](../../docs/adr/0002-no-absolute-rating.md) [ADR-0005](../../docs/adr/0005-strength-measurement.md)
- **Play (03… / ADR-0003):** terminal via a UCI bridge over the inference path; no custom GUI. [ADR-0003](../../docs/adr/0003-terminal-uci-play.md)
- **Compute envelope (ADR-0004):** strictly local, own hardware, $0; MPS allowed; background-friendly + turbo; ≤24h checkpointed. Reaffirmed and re-achievable on the M6 in ADR-0010.
- **Scope (05-chess-vs-tictactoe-scope, resolved):** the destination is real 8×8 chess; 3×3 is legacy. (On `alphazero` this is already true — the branch *is* the chess engine; `main` holds the legacy 3×3.)
- **Representation (06-chess-game-implementation, resolved):** 8-plane input, **133 actions** (64 quiet + 64 captures + 5 castles; promotions fold into destination squares), python-chess legality, termination on mate/stalemate/threefold/insufficient-material/150-halfmove. **Branch divergence:** the branch encodes 104 planes / 4672 actions with colliding promotion indices — ticket 08. [ADR-0006](../../docs/adr/0006-chess-representation.md)
- **Generation (07-vectorized-mcts, resolved):** fully vectorized MCTS on MPS-torch (flat numpy tree, batched PUCT, one batched net call/iteration). Measured at chess scale on **this machine**: unbatched 141 ms/move MPS / 83 ms CPU; naive batched-32 is 10× *worse* — the per-leaf Python floor is the bottleneck, backend-independent. ORT stays for eval + terminal play. **Branch divergence:** the branch's MCTS is the unbatched per-leaf form plus a queue-based BatchManager (the naive batching we rejected) — ticket 10. [ADR-0007](../../docs/adr/0007-vectorized-mcts-mps.md) Proto: `.scratch/elo-frontier/proto_chess_mcts.py`.
- **Data engine (03-self-play-data-engine, resolved):** concurrent gen+train on MPS with CPU fallback, live weights, progressive 25→100→250 sims, ~250k-move ring buffer, ONNX only at the end. **Branch divergence:** 8-ply stub trajectories with dummy 0.0 values, 1024-entry buffer, static weights per iteration — tickets 08/09. [ADR-0008](../../docs/adr/0008-data-engine.md)
- **Network scale (04-network-scale, resolved):** progressive A (f64/2) → B (f128/4); C (f256/8) rejected (≈2× slower generation, ≈20× slower training). Branch default is f128/b4 = B, but on 104 input planes; re-measure after the representation lands (ticket 08). [ADR-0009](../../docs/adr/0009-network-scale.md) Proto: `.scratch/elo-frontier/proto_net_scale.py`.
- **Branch + hardware pivot (resolved this session → ADR-0010):** work on `alphazero`; `main` frozen; Intel-2019→M6/Metal; local-first, credentials externalized. [ADR-0010](../../docs/adr/0010-working-branch-alphazero-m6.md)

## Not yet specified

- The cost-curve reporting format the effort lands with (build-phase choice; ADR-0005 fixes the numbers it plots).
- Whether the branch's existing MCTS+net learns anything at all: trajectories are capped at 8 plies with **dummy 0.0 values**, so the value head has never seen a real game outcome — unknown until ticket 08 lands.
- What the current `evaluate` (argmax net vs **random** opponent, 90% threshold) actually measures — a weak baseline; the ADR-0005 reference protocol is not implemented.
- How well the branch runs on MPS *before* vectorization investment — the device logic only knows cuda/cpu; ticket 09 measures.

## Fog

- [x] 01-generation-throughput (resolved → ADR-0001, main-branch measurements; chess-scale evidence in 07)
- [x] 02-strength-measurement-protocol (resolved → ADR-0005)
- [x] 03-self-play-data-engine (resolved → ADR-0008)
- [x] 04-network-scale (resolved → ADR-0009)
- [x] 05-chess-vs-tictactoe-scope (resolved; re-anchored by ADR-0010)
- [x] 06-chess-game-implementation (resolved → ADR-0006)
- [x] 07-vectorized-mcts (resolved → ADR-0007)
- [ ] 08-self-play-data-correctness — MoveMapper promotion collisions, silent illegal-move fallback, 8-ply stub games, dummy values, 1024 buffer (blocks 09, 10, 11, 13)
- [ ] 09-mps-local-port — device resolution → mps, torch 2.2 env, BatchManager/trainer/eval on MPS, local throughput measurement (blocks 10, 11, 13)
- [ ] 10-vectorized-mcts-implementation — apply ADR-0007 to `src/training/search.py` per the proto (blocked by 08, 09; blocks 13)
- [ ] 11-eval-and-uci — ADR-0005 reference-protocol eval + UCI terminal play per ADR-0003 (blocked by 08, 09; blocks 13)
- [ ] 12-local-first-cleanup — externalize/remove hardcoded AWS account + role; make cloud an opt-in flag (independent)
- [ ] 13-cost-curve-run — the ≤24h local training run with the cost-curve report the destination asks for (blocked by 08–11)
