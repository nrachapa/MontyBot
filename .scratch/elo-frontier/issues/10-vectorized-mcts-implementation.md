Type: implementation
Status: blocked
Blocked by: 08, 09
Label: wayfinder:build

# Vectorized MCTS per ADR-0007

Replace `src/training/search.py`'s per-leaf MCTS (batch-1 net call per leaf + queue-based `BatchManager` proxy = the "naive batching" measured 10× *worse* in the proto) with the design ADR-0007 decided: flat numpy tree, batched PUCT selection/backup, one batched net call per iteration, on MPS-torch.

Reference implementation: `.scratch/elo-frontier/proto_chess_mcts.py` (the ADR-0007 prototype). Port its architecture into `src/` behind the existing `SearchStrategy` interface; delete or bypass `batch_manager.py`'s queue proxy once the vectorized loop replaces it.

Acceptance: games/h on MPS ≥ proto parity at equal sims (median of ≥3); legal-move coverage identical (python-chess, ADR-0006); the 08-corrected self-play path feeds it unmodified.
