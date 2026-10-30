Type: implementation
Status: done
Label: wayfinder:build

# Vectorized MCTS per ADR-0007

Replace `src/training/search.py`'s per-leaf MCTS (batch-1 net call per leaf + queue-based `BatchManager` proxy = the "naive batching" measured 10× *worse* in the proto) with the design ADR-0007 decided: flat numpy tree, batched PUCT selection/backup, one batched net call per iteration, on MPS-torch.

Reference implementation: `.scratch/elo-frontier/proto_flat_mcts.py` (the ADR-0007 prototype). Landed as `VectorizedMCTS` in `src/training/search.py` (flat node pool, batched PUCT, one batched net call per iteration, in-worker inference — no `batch_manager.py` proxy); `MCTS` remains as the `search_strategy` override. `batch_manager.py` is now only exercised by the unbatched path.

Acceptance: games/h on MPS ≥ proto parity at equal sims (median of ≥3); legal-move coverage identical (python-chess, ADR-0006); the 08-corrected self-play path feeds it unmodified.
