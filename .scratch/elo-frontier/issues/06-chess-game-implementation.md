Type: grilling
Status: resolved
Blocked by: (none)
Label: wayfinder:grilling

# Real-chess Game implementation

## Question

`05` settled that real 8×8 chess is the destination. Decide the `Game` design: move encoding, input planes, terminal/draw rules, and move-generation source. Each choice scales the cost curve.

## Answer (resolved 2026-10-12, HITL — recorded in [ADR-0006](../../../docs/adr/0006-chess-representation.md))

1. **Action space: 189** — 64 squares × {quiet, capture} + 5 castles; fixed for every position (batching-friendly, 07 depends on it). Promotions encode as quiet/capture on the destination square.
2. **Representation: 8 planes** — 6 piece types × 2 sides + 2 castling-rights. Rejected: 15-plane AlphaZero (halfmove/en-passant/age only matter in long drawish games a day-budget never reaches) and 2-plane (castling is board-invisible).
3. **Move generation + draw rules: `python-chess`** (already in requirements, never imported until this build) — legality, castling, en passant, threefold, insufficient material. A move-gen bug poisons self-play data; correctness over its ~1 ms cost.
4. **Termination**: checkmate (±1), stalemate (0), threefold repetition (0, early cut — weak models *will* shuffle), insufficient material (0), 150-halfmove hard cap (0). The existing `get_winner()` 3×3 logic goes with the legacy game.

**Consequences for the map:** the representation change forces a new `AlphaZeroNet` shape and a from-scratch retrain (old 3×3 checkpoints are legacy). 03 (data engine) is unblocked on buffer/throughput design; 07 (vectorized MCTS) is now the critical path — its 189×8-leaf economics set the games-per-day number the whole effort lives on.
