Type: implementation
Status: ready
Blocked by: (none — first in the new chain)
Label: wayfinder:build

# Self-play data correctness (the branch's data is not learnable)

The branch's self-play produces garbage for a value head. Fix all four in one ticket — they are one data path:

1. **MoveMapper promotion collisions** (`src/training/game.py`): `to_index` drops the from/to squares for promotions (all queen promos → 4096) and `from_index` assumes 8th-rank. Options: (a) re-encode per ADR-0006 (133 actions — the design of record, and it shrinks the action head 35×), or (b) minimally fix the 4672 scheme (per-piece promotion slots indexed by square). ADR-0006 wins unless measurement says otherwise.
2. **Silent illegal-move fallback** in `apply_move`: `next(iter(b.legal_moves))` masks every encoding bug and pollutes trajectories. Must never fire; assert instead.
3. **8-ply stub trajectories** in `ParallelSelfPlay._play_game` ("minimal trajectory to keep runtime tiny") with **dummy 0.0 values**: the value head has no signal. Real full games, terminal result broadcast to the trajectory, per ADR-0008.
4. **Buffer 1024** → ~250k-move ring per ADR-0008.

Acceptance: a 20-game smoke run yields trajectories whose values are the true game outcome; `assert`-checked move round-trip (`from_index(to_index(m)) == m` on every legal move of every game); buffer holds ≥250k positions. Self-check per CODING_STANDARDS.md before any real run.
